#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch

from tensorboardX import SummaryWriter

from others.utils import rouge_results_to_str, test_rouge, tile
from translate.beam import GNMTGlobalScorer

import os
import sys

import torch.nn.functional as F
import numpy as np
from IPython import embed
from operator import add
from models.style_utils import to_var, top_k_logits
import pickle
import csv

from pytorch_transformers import BertTokenizer



SmallConst = 1e-15

def perturb_past(past, model, prev, args, classifier, good_index=None, stepsize=0.01, vocab_size=50259,
                 original_probs=None, accumulated_hidden=None, true_past=None, grad_norms=None,
                 encoder_hidden_states=None, encoder_attention_mask=None):
    window_length = args.window_length
    gm_scale, kl_scale = args.fusion_gm_scale, args.fusion_kl_scale
    one_hot_vectors = []
    for good_list in good_index:
        good_list = list(filter(lambda x: len(x) <= 1, good_list))
        good_list = torch.tensor(good_list).cuda()
        num_good = good_list.shape[0]
        one_hot_good = torch.zeros(num_good, vocab_size).cuda()
        one_hot_good.scatter_(1, good_list, 1)
        one_hot_vectors.append(one_hot_good)


    # Generate inital perturbed past
    past_perturb_orig = [(np.random.uniform(0.0, 0.0, p.shape).astype('float32'))
                         for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if args.decay:
        decay_mask = torch.arange(0., 1.0 + SmallConst, 1.0/(window_length))[1:]
    else:
        decay_mask = 1.0

    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, current_length, _ = past[0].shape

    if current_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple([window_length]) + tuple(
            past[0].shape[-1:])

        zeros_key_val_shape = tuple(past[0].shape[:-2]) + tuple([current_length - window_length]) + tuple(
            past[0].shape[-1:])

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask*ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).cuda()
    else:
        window_mask = torch.ones_like(past[0]).cuda()

    loss_per_iter = []
    for i in range(args.num_iterations):
        past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
        past_perturb = [to_var(p_, requires_grad=True) for p_ in past_perturb]

        perturbed_past = list(map(add, past, past_perturb))

        _, _, _, current_length, _ = past_perturb[0].shape

        # Compute hidden using perturbed past
        output = model(input_ids=prev,
                       encoder_hidden_states=encoder_hidden_states,
                       encoder_attention_mask=encoder_attention_mask,
                       past_key_values=perturbed_past)
        future_past = output.past_key_values
        hidden = output.hidden_states[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()

        # TODO: Check the layer-norm consistency of this with trained discriminator
        logits = output.logits
        logits = logits[:, -1, :]
        probabs = F.softmax(logits, dim=-1)
        loss = 0.0
        loss_list = []
        if args.loss_type == 1 or args.loss_type == 3:
            for one_hot_good in one_hot_vectors:
                good_logits = torch.mm(probabs, torch.t(one_hot_good))
                loss_word = good_logits
                loss_word = torch.sum(loss_word)
                loss_word = -torch.log(loss_word)
                #loss_word = torch.sum(loss_word) /torch.sum(one_hot_good)
                loss += loss_word
                loss_list.append(loss_word)
            #print('words', loss.data.cpu().numpy())

        if args.loss_type == 2 or args.loss_type == 3:
            ce_loss = torch.nn.CrossEntropyLoss()
            new_true_past = true_past
            for i in range(args.horizon_length):

                future_probabs = F.softmax(logits, dim=-1)  # Get softmax
                future_probabs = torch.unsqueeze(future_probabs, dim=1)

                _, new_true_past = model(future_probabs, past=new_true_past)
                future_hidden = model.hidden_states  # Get expected hidden states
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(future_hidden, dim=1)
                
            predicted_sentiment = classifier(new_accumulated_hidden / (current_length + 1 + args.horizon_length))

            label = torch.tensor([args.label_class], device='cuda', dtype=torch.long)
            discrim_loss = ce_loss(predicted_sentiment, label)
            #print('discrim', discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)


        kl_loss = 0.0
        if kl_scale > 0.0:
            p = (F.softmax(original_probs[:, -1, :], dim=-1))
            p = p + SmallConst * (p <= SmallConst).type(torch.FloatTensor).cuda().detach()
            correction = SmallConst * (probabs <= SmallConst).type(torch.FloatTensor).cuda().detach()
            corrected_probabs = probabs + correction.detach()
            kl_loss = kl_scale * ((corrected_probabs * (corrected_probabs / p).log()).sum())
            #print('kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss  # + discrim_loss

        #print((loss - kl_loss).data.cpu().numpy())
        
        #loss_per_iter.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=True)
        del loss
        torch.cuda.empty_cache()
        if grad_norms is not None and args.loss_type == 1:
            grad_norms = [torch.max(grad_norms[index], torch.norm(p_.grad * window_mask)) for index, p_ in
                          enumerate(past_perturb)]
        else:
            grad_norms = [(torch.norm(p_.grad * window_mask) + SmallConst) for index, p_ in enumerate(past_perturb)]

        grad = [
            -stepsize * (p_.grad * window_mask / grad_norms[index] ** args.gamma).data.cpu().numpy()
            for index, p_ in enumerate(past_perturb)]
        past_perturb_orig = list(map(add, grad, past_perturb_orig))

        for p_ in past_perturb:
            p_.grad.data.zero_()

        new_past = []
        for p in past:
            new_past.append(p.detach())

        past = new_past

    past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
    past_perturb = [to_var(p_, requires_grad=True) for p_ in past_perturb]
    perturbed_past = list(map(add, past, past_perturb))

    return perturbed_past, new_accumulated_hidden, grad_norms, loss_per_iter


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        self.global_scorer = global_scorer
        #self.beam_size = args.beam_size
        self.beam_size = 1
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}
            
        self.vocab_bert = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.tgt_str, batch.src

        translations = []
        for b in range(batch_size):
            pred_sents = []
            for n in preds[b][0]:
                pred_sents.append(self.vocab.decode(int(n)).replace(' ', ''))
            pred_sents = ' '.join(pred_sents).replace(' ##','')
            print('pred_sents : ')
            print(pred_sents)
            gold_sent = ' '.join(tgt_str[b].split())
            # translation = Translation(fname[b],src[:, b] if src is not None else None,
            #                           src_raw, pred_sents,
            #                           attn[b], pred_score[b], gold_sent,
            #                           gold_score[b])
            # src = self.spm.DecodeIds([int(t) for t in translation_batch['batch'].src[0][5] if int(t) != len(self.spm)])
            raw_src = [self.vocab_bert.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            translation = (pred_sents, gold_sent, raw_src)
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        # raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        # raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0
        #with torch.no_grad():
        for batch in data_iter:
            if(self.args.recall_eval):
                gold_tgt_len = batch.tgt.size(1)
                self.min_length = gold_tgt_len + 20
                self.max_length = gold_tgt_len + 60
            batch_data = self.translate_batch(batch)
            translations = self.from_batch(batch_data)

            for trans in translations:
                pred, gold, src = trans
                pred_str = pred.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
                gold_str = gold.strip()
                if(self.args.recall_eval):
                    _pred_str = ''
                    gap = 1e3
                    for sent in pred_str.split('<q>'):
                        can_pred_str = _pred_str+ '<q>'+sent.strip()
                        can_gap = math.fabs(len(_pred_str.split())-len(gold_str.split()))
                        # if(can_gap>=gap):
                        if(len(can_pred_str.split())>=len(gold_str.split())+10):
                            pred_str = _pred_str
                            break
                        else:
                            gap = can_gap
                            _pred_str = can_pred_str



                    # pred_str = ' '.join(pred_str.split()[:len(gold_str.split())])
                # self.raw_can_out_file.write(' '.join(pred).strip() + '\n')
                # self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                self.can_out_file.write(pred_str + '\n')
                self.gold_out_file.write(gold_str + '\n')
                self.src_out_file.write(src.strip() + '\n')
                ct += 1
            self.can_out_file.flush()
            self.gold_out_file.flush()
            self.src_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        if (step != -1):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        #with torch.no_grad():
        return self._fast_translate_batch(
            batch,
            self.max_length,
            min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src

        src_features = self.model.bert(src, segs, mask_src)
        #dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        """
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        """
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch
        
        
        def list_tokens(word_list, vocab):
            token_list = []
            for word in word_list:
                token_list.append(vocab.encode(" " + word))
            return token_list
        
        good_index = []
        if self.args.bag_of_words:
            bags_of_words = self.args.bag_of_words.split(";")
            for wordlist in bags_of_words:
                with open(wordlist, "r") as f:
                    words = f.read()
                    words = words.split('\n')
                good_index.append(list_tokens(words, self.vocab))
        
        past = None
        grad_norms = None

        if not self.args.perturb:
            for step in range(max_length):
                with torch.no_grad():
                    decoder_input = alive_seq[:, -1].view(1, -1)

                    # Decoder forward.
                    decoder_input = decoder_input.transpose(0,1)
                    """
                    dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
                                                            step=step)
                    """
                    output = self.model.decoder(input_ids=decoder_input,
                                                encoder_hidden_states=src_features,
                                                encoder_attention_mask=mask_src,
                                                past_key_values=past)
                    dec_out = output.hidden_states[-1]
                    past = output.past_key_values                    
                    # Generator forward.
                    log_probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))
                    vocab_size = log_probs.size(-1)

                    if step < min_length:
                        log_probs[:, self.end_token] = -1e20

                    # Multiply probs by the beam probability.
                    log_probs += topk_log_probs.view(-1).unsqueeze(1)

                    alpha = self.global_scorer.alpha
                    length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

                    # Flatten probs into a list of possibilities.
                    curr_scores = log_probs / length_penalty

                    if(self.args.block_trigram):
                        cur_len = alive_seq.size(1)
                        if(cur_len>3):
                            for i in range(alive_seq.size(0)):
                                fail = False
                                seq = [int(w) for w in alive_seq[i]]
                                words = []
                                for w in seq:
                                    words.append(self.vocab.decode(w))
                                words = ' '.join(words).replace(' ##','').split()
                                if(len(words)<=3):
                                    continue
                                trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                                trigram = tuple(trigrams[-1])
                                if trigram in trigrams[:-1]:
                                    fail = True
                                if fail:
                                    curr_scores[i] = -10e20

                    curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
                    topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

                    # Recover log probs.
                    topk_log_probs = topk_scores * length_penalty

                    # Resolve beam origin and true word ids.
                    topk_beam_index = topk_ids.div(vocab_size)
                    topk_ids = topk_ids.fmod(vocab_size)

                    # Map beam_index to batch_index in the flat representation.
                    batch_index = (
                            topk_beam_index
                            + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
                    select_indices = batch_index.view(-1)

                    # Append last prediction.
                    alive_seq = torch.cat(
                        [alive_seq.index_select(0, select_indices),
                        topk_ids.view(-1, 1)], -1)

                    is_finished = topk_ids.eq(self.end_token)
                    if step + 1 == max_length:
                        is_finished.fill_(1)
                    # End condition is top beam is finished.
                    end_condition = is_finished[:, 0].eq(1)
                    # Save finished hypotheses.
                    if is_finished.any():
                        predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                        for i in range(is_finished.size(0)):
                            b = batch_offset[i]
                            if end_condition[i]:
                                is_finished[i].fill_(1)
                            finished_hyp = is_finished[i].nonzero().view(-1)
                            # Store finished hypotheses for this batch.
                            for j in finished_hyp:
                                hypotheses[b].append((
                                    topk_scores[i, j],
                                    predictions[i, j, 1:]))
                            # If the batch reached the end, save the n_best hypotheses.
                            if end_condition[i]:
                                best_hyp = sorted(
                                    hypotheses[b], key=lambda x: x[0], reverse=True)
                                score, pred = best_hyp[0]

                                results["scores"][b].append(score)
                                results["predictions"][b].append(pred)
                        non_finished = end_condition.eq(0).nonzero().view(-1)
                        # If all sentences are translated, no need to go further.
                        if len(non_finished) == 0:
                            break
                        # Remove finished batches for the next step.
                        topk_log_probs = topk_log_probs.index_select(0, non_finished)
                        batch_index = batch_index.index_select(0, non_finished)
                        batch_offset = batch_offset.index_select(0, non_finished)
                        alive_seq = predictions.index_select(0, non_finished) \
                            .view(-1, alive_seq.size(-1))
                    # Reorder states.
                    select_indices = batch_index.view(-1)
                    src_features = src_features.index_select(0, select_indices)
                    """
                    dec_states.map_batch_fn(
                        lambda state, dim: state.index_select(dim, select_indices))
                    """
                    mask_src = mask_src.index_select(0, select_indices)
                    past_copy = past
                    past = ()
                    for pp in past_copy:
                        pp = pp.index_select(1, select_indices)
                        past += (pp,)
            else:
                for step in range(max_length):
                    with torch.no_grad():
                        if past is None:
                            prev = alive_seq[:,-1:]
                            if alive_seq.shape[1] > 1:
                                output = self.model.decoder(input_ids=alive_seq[:,:-1],
                                                            encoder_hidden_states=src_features,
                                                            encoder_attention_mask=mask_src,
                                                            past_key_values=past)
                                past = output.past_key_values
                        output_original = self.model.decoder(input_ids=alive_seq,
                                                            encoder_hidden_states=src_features,
                                                            encoder_attention_mask=mask_src,
                                                            past_key_values=past)
                        original_probs, true_past = output_original.logits, output_original.past_key_values
                        true_hidden = output_original.hidden_states[-1]
                        
                        if step >= self.args.grad_length:
                            current_stepsize = self.args.stepsize * 0
                        else:
                            current_stepsize = self.args.stepsize
                        
                        accumulated_hidden = output_original.hidden_states[:, :-1, :]
                        accumulated_hidden = torch.sum(accumulated_hidden, dim=1)
                        
                    perturbed_past, _, grad_norms, loss_per_iter = perturb_past(past=past,
                                                                                model=self.model.decoder,
                                                                                prev=prev,
                                                                                args=self.args,
                                                                                good_index=good_index, 
                                                                                stepsize=current_stepsize,
                                                                                original_probs=original_probs,
                                                                                true_past=true_past,
                                                                                accumulated_hidden=accumulated_hidden,
                                                                                classifier=None,
                                                                                grad_norms=grad_norms,
                                                                                encoder_hidden_states=src_features,
                                                                                encoder_attention_mask=mask_src)
                    with torch.no_grad():
                        output_test = self.model.decoder(input_ids=prev,
                                                         encoder_hidden_states=src_features,
                                                         encoder_attention_mask=mask_src,
                                                         past_key_values=perturbed_past)
                        test_logits, past = output_test.logits, output_test.past_key_values
                        
                        logits = test_logits[:, -1, :] / self.args.temperature
                        
                        log_probs = F.softmax(logits, dim=-1)
                        
                        original_probs = F.softmax(original_probs[:, -1, :], dim=-1)
                        
                        gm_scale = self.args.fusion_gm_scale
                        log_probs = ((log_probs ** gm_scale) * (original_probs ** (1 - gm_scale)))  # + SmallConst

                        

                        if torch.sum(log_probs) <= 1:
                            log_probs = log_probs / torch.sum(log_probs)
                        
                        # Generator forward.
                        log_probs = torch.log(log_probs)
                        vocab_size = log_probs.size(-1)

                        if step < min_length:
                            log_probs[:, self.end_token] = -1e20

                        # Multiply probs by the beam probability.
                        log_probs += topk_log_probs.view(-1).unsqueeze(1)

                        alpha = self.global_scorer.alpha
                        length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

                        # Flatten probs into a list of possibilities.
                        curr_scores = log_probs / length_penalty

                        if(self.args.block_trigram):
                            cur_len = alive_seq.size(1)
                            if(cur_len>3):
                                for i in range(alive_seq.size(0)):
                                    fail = False
                                    seq = [int(w) for w in alive_seq[i]]
                                    words = []
                                    for w in seq:
                                        words.append(self.vocab.decode(w).replace(' ',''))
                                    words = ' '.join(words).replace(' ##','').split()
                                    if(len(words)<=3):
                                        continue
                                    trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                                    trigram = tuple(trigrams[-1])
                                    if trigram in trigrams[:-1]:
                                        fail = True
                                    if fail:
                                        curr_scores[i] = -10e20

                        curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
                        topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

                        # Recover log probs.
                        topk_log_probs = topk_scores * length_penalty

                        # Resolve beam origin and true word ids.
                        topk_beam_index = topk_ids.div(vocab_size)
                        topk_ids = topk_ids.fmod(vocab_size)
                        
                        prev = topk_ids

                        # Map beam_index to batch_index in the flat representation.
                        batch_index = (
                                topk_beam_index
                                + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
                        select_indices = batch_index.view(-1)

                        # Append last prediction.
                        alive_seq = torch.cat(
                            [alive_seq.index_select(0, select_indices),
                            topk_ids.view(-1, 1)], -1)

                        is_finished = topk_ids.eq(self.end_token)
                        if step + 1 == max_length:
                            is_finished.fill_(1)
                        # End condition is top beam is finished.
                        end_condition = is_finished[:, 0].eq(1)
                        # Save finished hypotheses.
                        if is_finished.any():
                            predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                            for i in range(is_finished.size(0)):
                                b = batch_offset[i]
                                if end_condition[i]:
                                    is_finished[i].fill_(1)
                                finished_hyp = is_finished[i].nonzero().view(-1)
                                # Store finished hypotheses for this batch.
                                for j in finished_hyp:
                                    hypotheses[b].append((
                                        topk_scores[i, j],
                                        predictions[i, j, 1:]))
                                # If the batch reached the end, save the n_best hypotheses.
                                if end_condition[i]:
                                    best_hyp = sorted(
                                        hypotheses[b], key=lambda x: x[0], reverse=True)
                                    score, pred = best_hyp[0]

                                    results["scores"][b].append(score)
                                    results["predictions"][b].append(pred)
                            non_finished = end_condition.eq(0).nonzero().view(-1)
                            # If all sentences are translated, no need to go further.
                            if len(non_finished) == 0:
                                break
                            # Remove finished batches for the next step.
                            topk_log_probs = topk_log_probs.index_select(0, non_finished)
                            batch_index = batch_index.index_select(0, non_finished)
                            batch_offset = batch_offset.index_select(0, non_finished)
                            alive_seq = predictions.index_select(0, non_finished) \
                                .view(-1, alive_seq.size(-1))
                        # Reorder states.
                        select_indices = batch_index.view(-1)
                        prev = prev.index_select(0, select_indices)
                        src_features = src_features.index_select(0, select_indices)
                        mask_src = mask_src.index_select(0, select_indices)
                        past_copy = past
                        past = ()
                        for pp in past_copy:
                            pp = pp.index_select(1, select_indices)
                            past += (pp,)
                

        return results


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
