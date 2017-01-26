# Author: Daniel Ortiz Mart\'inez
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import re
import sys
from heapq import heappop
from heapq import heappush

from thot_utils.libs import config
from thot_utils.libs.utils import is_alnum
from thot_utils.libs.utils import is_categ
from thot_utils.libs.utils import is_number
from thot_utils.libs.utils import split_string_to_words
from thot_utils.libs.utils import transform_word


class TransModel(object):
    def __init__(self, model_provider):
        self.model_provider = model_provider

    def obtain_opts_for_src(self, src_words):
        return self.model_provider.get_targets(src_words)

    def obtain_srctrg_count(self, src_words, trg_words):
        return self.model_provider.get_target_count(src_words, trg_words)

    def obtain_trgsrc_prob(self, src_words, trg_words):
        sc = self.obtain_src_count(src_words)
        if sc == 0:
            return 0
        else:
            stc = self.obtain_srctrg_count(src_words, trg_words)
            return float(stc) / float(sc)

    def obtain_trgsrc_prob_smoothed(self, src_words, trg_words):
        sc = self.obtain_src_count(src_words)
        if sc == 0:
            return config.tm_smooth_prob
        else:
            stc = self.obtain_srctrg_count(src_words, trg_words)
            return (1 - config.tm_smooth_prob) * (float(stc) / float(sc))

    def obtain_src_count(self, src_words):
        return self.model_provider.get_source_count(src_words)

    def get_mon_hyp_state(self, hyp):
        if len(hyp.data.coverage) == 0:
            return 0
        else:
            return hyp.data.coverage[len(hyp.data.coverage) - 1]


class LangModel:
    def __init__(self, provider, ngrams_length, interp_prob=None):
        self.provider = provider
        self.ngrams_length = ngrams_length
        self.set_interp_prob(interp_prob or config.lm_interp_prob)

    def set_interp_prob(self, interp_prob):
        if interp_prob > 0.99:
            self.interp_prob = 0.99
        elif interp_prob < 0:
            self.interp_prob = 0
        else:
            self.interp_prob = interp_prob

    def obtain_ng_count(self, ngram):
        return self.provider.get_count(ngram)

    def obtain_trgsrc_prob(self, ngram):
        if ngram == "":
            return 1.0 / self.obtain_ng_count("")
        else:
            hc = self.obtain_ng_count(self.remove_newest_word(ngram))
            if hc == 0:
                return 0
            else:
                ngc = self.obtain_ng_count(ngram)
                return ngc / hc

    def obtain_trgsrc_interp_prob(self, ngram):
        ng_array = ngram.split()
        if len(ng_array) == 0:
            return self.obtain_trgsrc_prob(ngram)
        else:
            return (
                self.interp_prob * self.obtain_trgsrc_prob(ngram) +
                (1 - self.interp_prob) * self.obtain_trgsrc_interp_prob(self.remove_oldest_word(ngram))
            )

    def remove_newest_word(self, ngram):
        ng_array = ngram.split()
        if len(ng_array) <= 1:
            return ""
        else:
            result = ng_array[0]
            for i in range(1, len(ng_array) - 1):
                result = result + " " + ng_array[i]
            return result

    def remove_oldest_word(self, ngram):
        ng_array = ngram.split()
        if len(ng_array) <= 1:
            return ""
        else:
            result = ng_array[1]
            for i in range(2, len(ng_array)):
                result = result + " " + ng_array[i]
            return result

    def lm_preproc(self, trans_raw_word_array, lmvoc):
        # Do not alter words
        return trans_raw_word_array

    def get_lm_state(self, words):
        # Obtain array of previous words including BOS symbol
        words_array_aux = words.split()
        words_array = []
        words_array.append(config.bos_str)
        for i in range(len(words_array_aux)):
            words_array.append(words_array_aux[i])

        # Obtain history from array
        len_hwa = len(words_array)
        hist = ""
        for i in range(self.ngrams_length - 1):
            if i < len(words_array):
                word = words_array[len_hwa - 1 - i]
                if hist == "":
                    hist = word
                else:
                    hist = word + " " + hist
        return hist

    def get_hyp_state(self, hyp):
        return self.get_lm_state(hyp.data.words)


class BfsHypdata:
    def __init__(self):
        self.coverage = []
        self.words = ""

    def __str__(self):
        result = "cov:"
        for k in range(len(self.coverage)):
            result = result + " " + str(self.coverage[k])
        result = result + " ; words: " + self.words.encode("utf-8")
        return result


class Hypothesis:
    def __init__(self):
        self.score = 0
        self.data = BfsHypdata()

    def __lt__(self, other):
        # by default we want to have hypothesis with higher score at the beginning
        return self.score > other.score


class PriorityQueue:
    def __init__(self):
        self.heap = []

    def empty(self):
        return len(self.heap) == 0

    def put(self, item):
        heappush(self.heap, item)

    def get(self):
        return heappop(self.heap)


class StateInfoDict:
    def __init__(self):
        self.recomb_map = {}

    def empty(self):
        return len(self.recomb_map) == 0

    def insert(self, state_info, score):
        # Update recombination info
        if state_info in self.recomb_map:
            if score > self.recomb_map[state_info]:
                self.recomb_map[state_info] = score
        else:
            self.recomb_map[state_info] = score

    def get(self):
        return heappop(self.heap)

    def hyp_recombined(self, state_info, score):

        if state_info in self.recomb_map:
            if score < self.recomb_map[state_info]:
                return True
            else:
                return False
        else:
            return False


class StateInfo:
    def __init__(self, tm_state, lm_state):
        self.tm_state = tm_state
        self.lm_state = lm_state

    def __hash__(self):
        return hash((self.tm_state, self.lm_state))

    def __eq__(self, other):
        return (self.tm_state, self.lm_state) == (other.tm_state, other.lm_state)


def obtain_state_info(tmodel, lmodel, hyp):
    return StateInfo(tmodel.get_mon_hyp_state(hyp), lmodel.get_hyp_state(hyp))


def areCategsPaired(categ_src_ann_words_array, categ_trg_ann_words_array):
    # Initialize dictionaries with number of category ocurrences
    num_categs_src = {}
    num_categs_trg = {}
    for categ in config.categ_set:
        num_categs_src[categ] = 0
        num_categs_trg[categ] = 0
        # Count source categories
    for i in range(len(categ_src_ann_words_array)):
        if categ_src_ann_words_array[i] in config.categ_set:
            num_categs_src[categ_src_ann_words_array[i]] += 1
    # Count target categories
    for i in range(len(categ_trg_ann_words_array)):
        if categ_trg_ann_words_array[i] in config.categ_set:
            num_categs_trg[categ_trg_ann_words_array[i]] += 1
    # Verify category ocurrence number equality
    for categ in num_categs_src:
        if not num_categs_src[categ] == num_categs_trg[categ]:
            return False

    return True


##################################################
def categ_src_trg_annotation(src_ann_words, trg_ann_words):
    # Obtain array with source words (with a without categorization)
    src_ann_words_array = src_ann_words.split()
    categ_src_ann_words_array = []
    for i in range(len(src_ann_words_array)):
        categ_src_ann_words_array.append(categorize_word(src_ann_words_array[i]))

    # Obtain array with target words (with a without categorization)
    trg_ann_words_array = trg_ann_words.split()
    categ_trg_ann_words_array = []
    for i in range(len(trg_ann_words_array)):
        categ_trg_ann_words_array.append(categorize_word(trg_ann_words_array[i]))

    # Verify that categories are paired
    if areCategsPaired(categ_src_ann_words_array, categ_trg_ann_words_array):
        return categ_src_ann_words_array, categ_trg_ann_words_array
    else:
        return src_ann_words_array, trg_ann_words_array


##################################################
def categorize(sentence):
    skeleton = list(annotated_string_to_xml_skeleton(sentence))

    # Categorize words
    categ_word_array = []
    curr_xml_tag = None
    for i in range(len(skeleton)):
        is_tag, word = skeleton[i]
        if is_tag:
            # Treat xml tag
            if word == '<' + config.len_ann + '>':
                categ_word_array.append(word)
                curr_xml_tag = "len_ann"
            elif word == '<' + config.grp_ann + '>':
                categ_word_array.append(word)
            elif word == '<' + config.src_ann + '>':
                curr_xml_tag = "src_ann"
            elif word == '<' + config.trg_ann + '>':
                curr_xml_tag = "trg_ann"
            elif word == '</' + config.len_ann + '>':
                categ_word_array.append(word)
                curr_xml_tag = None
            elif word == '</' + config.grp_ann + '>':
                categ_word_array.append(word)
            elif word == '</' + config.src_ann + '>':
                curr_xml_tag = None
            elif word == '</' + config.trg_ann + '>':
                curr_xml_tag = None
                categ_src_words, categ_trg_words = categ_src_trg_annotation(src_ann_words, trg_ann_words)
                # Add source phrase
                categ_word_array.append('<' + config.src_ann + '>')
                for i in range(len(categ_src_words)):
                    categ_word_array.append(categ_src_words[i])
                categ_word_array.append('</' + config.src_ann + '>')
                # Add target phrase
                categ_word_array.append('<' + config.trg_ann + '>')
                for i in range(len(categ_trg_words)):
                    categ_word_array.append(categ_trg_words[i])
                categ_word_array.append('</' + config.trg_ann + '>')
        else:
            # Categorize group of words
            if curr_xml_tag is None:
                word_array = word.split()
                for j in range(len(word_array)):
                    categ_word_array.append(categorize_word(word_array[j]))
            elif curr_xml_tag == "len_ann":
                word_array = word.split()
                for j in range(len(word_array)):
                    categ_word_array.append(word)
            elif curr_xml_tag == "src_ann":
                src_ann_words = word
            elif curr_xml_tag == "trg_ann":
                trg_ann_words = word

    return u' '.join(categ_word_array)


def categorize_word(word):
    if word.isdigit():
        if len(word) > 1:
            return config.number_str
        else:
            return config.digit_str
    elif is_number(word):
        return config.number_str
    elif is_alnum(word) and config.digits.search(word):
        return config.alfanum_str
    else:
        return word


def extract_alig_info(hyp_word_array):
    # Initialize output variables
    srcsegms = []
    trgcuts = []

    # Scan hypothesis information
    info_found = False
    for i in range(len(hyp_word_array)):
        if hyp_word_array[i] == "hypkey:" and hyp_word_array[i - 1] == "|":
            info_found = True
            i -= 2
            break

    if info_found:
        # Obtain target segment cuts
        trgcuts_found = False
        while i > 0:
            if hyp_word_array[i] != "|":
                trgcuts.append(int(hyp_word_array[i]))
                i -= 1
            else:
                trgcuts_found = True
                i -= 1
                break
        trgcuts.reverse()

        if trgcuts_found:
            # Obtain source segments
            srcsegms_found = False
            while i > 0:
                if hyp_word_array[i] != "|":
                    if i > 3:
                        srcsegms.append((int(hyp_word_array[i - 3]), int(hyp_word_array[i - 1])))
                    i -= 5
                else:
                    srcsegms_found = True
                    break
            srcsegms.reverse()

    # Return result
    if srcsegms_found:
        return srcsegms, trgcuts
    else:
        return [], []


def extract_categ_words_of_segm(word_array, left, right):
    # Initialize variables
    categ_words = []

    # Explore word array
    for i in range(left, right + 1):
        if is_categ(word_array[i]) or is_categ(categorize_word(word_array[i])):
            categ_words.append((i, word_array[i]))

    # Return result
    return categ_words


def decategorize(sline, tline, iline):
    src_word_array = sline.split()
    trg_word_array = tline.split()
    hyp_word_array = iline.split()

    # Extract alignment information
    srcsegms, trgcuts = extract_alig_info(hyp_word_array)

    # Iterate over target words
    output = ""
    for trgpos in range(len(trg_word_array)):

        if is_categ(trg_word_array[trgpos]):
            output += decategorize_word(trgpos, src_word_array, trg_word_array, srcsegms, trgcuts)
        else:
            output += trg_word_array[trgpos]

        if trgpos < len(trg_word_array) - 1:
            output += " "

    return output


def decategorize_word(trgpos, src_word_array, trg_word_array, srcsegms, trgcuts):
    # Check if there is alignment information available
    if len(srcsegms) == 0 or len(trgcuts) == 0:
        return trg_word_array
    else:
        # Scan target cuts
        for k in range(len(trgcuts)):
            if k == 0:
                if trgpos + 1 <= trgcuts[k]:
                    trgleft = 0
                    trgright = trgcuts[k] - 1
                    break
            else:
                if trgpos + 1 > trgcuts[k - 1] and trgpos + 1 <= trgcuts[k]:
                    trgleft = trgcuts[k - 1]
                    trgright = trgcuts[k] - 1
                    break
        # Check if trgpos'th word was assigned to one cut
        if k < len(trgcuts):
            # Obtain source segment limits
            srcleft = srcsegms[k][0] - 1
            srcright = srcsegms[k][1] - 1
            # Obtain categorized words with their indices
            src_categ_words = extract_categ_words_of_segm(src_word_array, srcleft, srcright)
            trg_categ_words = extract_categ_words_of_segm(trg_word_array, trgleft, trgright)

            # Obtain decategorized word
            decateg_word = ""
            curr_categ_word = trg_word_array[trgpos]
            curr_categ_word_order = 0
            for l in range(len(trg_categ_words)):
                if trg_categ_words[l][0] == trgpos:
                    break
                else:
                    if trg_categ_words[l][1] == curr_categ_word:
                        curr_categ_word_order += 1

            aux_order = 0
            for l in range(len(src_categ_words)):
                if categorize_word(src_categ_words[l][1]) == curr_categ_word:
                    if aux_order == curr_categ_word_order:
                        decateg_word = src_categ_words[l][1]
                        break
                    else:
                        aux_order += 1

            # Return decategorized word
            if decateg_word == "":
                return trg_word_array[trgpos]
            else:
                return decateg_word
        else:
            return trg_word_array[trgpos]


class Decoder:
    def __init__(self, tmodel, lmodel, weights):
        # Initialize data members
        self.tmodel = tmodel
        self.lmodel = lmodel
        self.weights = weights

        # Checking on weight list
        if len(self.weights) != 4:
            self.weights = [1, 1, 1, 1]
        else:
            print("Decoder weights:", file=sys.stderr)
            for i in range(len(weights)):
                print(weights[i], file=sys.stderr)
            print("", file=sys.stderr)

        # Set indices for weight list
        self.tmw_idx = 0
        self.phrpenw_idx = 1
        self.wpenw_idx = 2
        self.lmw_idx = 3

    def opt_contains_src_words(self, src_words, opt):

        st = ""
        src_words_array = src_words.split()
        for i in range(len(src_words_array)):
            st = st + src_words_array[i]

        if st == opt:
            return True
        else:
            return False

    def tm_ext_lp(self, new_src_words, opt, verbose):

        lp = math.log(self.tmodel.obtain_trgsrc_prob_smoothed(new_src_words, opt))

        if verbose:
            print(
                "  tm: logprob(", opt.encode("utf-8"), "|", new_src_words.encode("utf-8"), ")=", lp,
                file=sys.stderr
            )

        return lp

    def pp_ext_lp(self, verbose):
        lp = math.log(1.0 / math.e)
        if verbose:
            print("  pp:", lp, file=sys.stderr)
        return lp

    def wp_ext_lp(self, words, verbose):

        nw = len(words.split())

        lp = nw * math.log(1 / math.e)

        if verbose:
            print("  wp:", lp, file=sys.stderr)

        return lp

    def lm_transform_word(self, word):
        # Do not alter word
        return word

    def lm_transform_word_unk(self, word):
        # Introduce unknown word
        if self.lmodel.obtain_ng_count(word) == 0:
            return config.unk_word_str
        else:
            return word

    def lm_ext_lp(self, hyp_words, opt, verbose):
        # Obtain lm history
        rawhist = self.lmodel.get_lm_state(hyp_words)
        rawhist_array = rawhist.split()
        hist = ""
        for i in range(len(rawhist_array)):
            word = self.lm_transform_word(rawhist_array[i])
            if hist == "":
                hist = word
            else:
                hist = hist + " " + word

        # Obtain logprob for new words
        lp = 0
        opt_words_array = opt.split()
        for i in range(len(opt_words_array)):
            word = self.lm_transform_word(opt_words_array[i])
            if hist == "":
                ngram = word
            else:
                ngram = hist + " " + word
            lp_ng = math.log(self.lmodel.obtain_trgsrc_interp_prob(ngram))
            lp += lp_ng
            if verbose:
                print("  lm: logprob(", word.encode("utf-8"), "|", hist.encode("utf-8"), ")=", lp_ng, file=sys.stderr)

            hist = self.lmodel.remove_oldest_word(ngram)

        return lp

    def expand(self, tok_array, hyp, new_hyp_cov, verbose):
        # Init result
        exp_list = []

        # Obtain words to be translated
        new_src_words = ""
        last_cov_pos = self.last_cov_pos(hyp.data.coverage)
        for i in range(last_cov_pos + 1, new_hyp_cov + 1):
            if new_src_words == "":
                new_src_words = tok_array[i]
            else:
                new_src_words = new_src_words + " " + tok_array[i]

        # Obtain translation options
        opt_list = self.tmodel.obtain_opts_for_src(new_src_words)

        # If there are no options and only one source word is being covered,
        # artificially add one
        if len(opt_list) == 0 and len(new_src_words.split()) == 1:
            opt_list.append(new_src_words)

        # Print information about expansion if in verbose mode
        if verbose:
            print(
                "++ expanding -> new_hyp_cov:", new_hyp_cov, "; new_src_words:", new_src_words.encode("utf-8"),
                "; num options:", len(opt_list), file=sys.stderr
            )

        # Iterate over options
        for opt in opt_list:

            if verbose:
                print("   option:", opt.encode("utf-8"), file=sys.stderr)

            # Extend hypothesis

            # Obtain new hypothesis
            bfsd_newhyp = BfsHypdata()

            # Obtain coverage for new hyp
            bfsd_newhyp.coverage = hyp.data.coverage[:]
            bfsd_newhyp.coverage.append(new_hyp_cov)

            # Obtain list of words for new hyp
            if hyp.data.words == "":
                bfsd_newhyp.words = opt
            else:
                bfsd_newhyp.words = hyp.data.words
                bfsd_newhyp.words = bfsd_newhyp.words + " " + opt

            # Obtain score for new hyp

            # Add translation model contribution
            tm_lp = self.tm_ext_lp(new_src_words, opt, verbose)
            w_tm_lp = self.weights[self.tmw_idx] * tm_lp

            # Add phrase penalty contribution
            pp_lp = self.pp_ext_lp(verbose)
            w_pp_lp = self.weights[self.phrpenw_idx] * pp_lp

            # Add word penalty contribution
            wp_lp = self.wp_ext_lp(opt, verbose)
            w_wp_lp = self.weights[self.wpenw_idx] * wp_lp

            # Add language model contribution
            lm_lp = self.lm_ext_lp(hyp.data.words, opt, verbose)
            w_lm_lp = self.weights[self.lmw_idx] * lm_lp

            # Add language model contribution for <bos> if hyp is
            # complete
            w_lm_end_lp = 0
            if self.cov_is_complete(bfsd_newhyp.coverage, tok_array):
                lm_end_lp = self.lm_ext_lp(bfsd_newhyp.words, config.eos_str, verbose)
                w_lm_end_lp = self.weights[self.lmw_idx] * lm_end_lp

            if verbose:
                print(
                    "   expansion ->", "w. lp:", hyp.score + w_tm_lp + w_pp_lp + w_lm_lp + w_lm_end_lp,
                    "; w. tm logprob:", w_tm_lp,
                    "; w. pp logprob:", w_pp_lp, "; w. wp logprob:", w_wp_lp, "; w. lm logprob:", w_lm_lp,
                    "; w. lm end logprob:", w_lm_end_lp, ";", str(bfsd_newhyp), file=sys.stderr)
                print("   ----", file=sys.stderr)

            # Obtain new hypothesis
            newhyp = Hypothesis()
            newhyp.score = hyp.score + w_tm_lp + w_pp_lp + w_wp_lp + w_lm_lp + w_lm_end_lp
            newhyp.data = bfsd_newhyp

            # Add expansion to list
            exp_list.append(newhyp)

        # Return result
        return exp_list

    def last_cov_pos(self, coverage):

        if len(coverage) == 0:
            return -1
        else:
            return coverage[len(coverage) - 1]

    def hyp_is_complete(self, hyp, src_word_array):

        return self.cov_is_complete(hyp.data.coverage, src_word_array)

    def cov_is_complete(self, coverage, src_word_array):

        if self.last_cov_pos(coverage) == len(src_word_array) - 1:
            return True
        else:
            return False

    def obtain_nblist(self, src_word_array, nblsize, verbose):
        # Insert initial hypothesis in stack
        priority_queue = PriorityQueue()
        hyp = Hypothesis()
        priority_queue.put(hyp)

        # Create state dictionary
        stdict = StateInfoDict()
        stdict.insert(obtain_state_info(self.tmodel, self.lmodel, hyp), hyp.score)
        stdict.insert(obtain_state_info(self.tmodel, self.lmodel, hyp), hyp.score)

        # Obtain n-best hypotheses
        nblist = []
        for i in range(nblsize):
            hyp = self.best_first_search(src_word_array, priority_queue, stdict, verbose)

            # Append hypothesis to nblist
            if len(hyp.data.coverage) > 0:
                nblist.append(hyp)

        # return result
        return nblist

    def obtain_detok_sent(self, tok_array, best_hyp):

        # Check if tok_array is not empty
        if len(tok_array) > 0:
            # Init variables
            result = ""
            coverage = best_hyp.data.coverage
            # Iterate over hypothesis coverage array
            for i in range(len(coverage)):
                # Obtain leftmost source position
                if i == 0:
                    leftmost_src_pos = 0
                else:
                    leftmost_src_pos = coverage[i - 1] + 1

                # Obtain detokenized word
                detok_word = ""
                for j in range(leftmost_src_pos, coverage[i] + 1):
                    detok_word = detok_word + tok_array[j]

                # Incorporate detokenized word to detokenized sentence
                if i == 0:
                    result = detok_word
                else:
                    result = result + " " + detok_word
            # Return detokenized sentence
            return result
        else:
            return ""

    def get_hypothesis_to_expand(self, priority_queue, stdict):

        while True:
            if priority_queue.empty():
                return True, Hypothesis()
            else:
                hyp = priority_queue.get()
                sti = obtain_state_info(self.tmodel, self.lmodel, hyp)
                if not stdict.hyp_recombined(sti, hyp.score):
                    return False, hyp

    def best_first_search(self, src_word_array, priority_queue, stdict, verbose):
        # Initialize variables
        end = False
        niter = 0

        if verbose:
            print("*** Starting best first search...", file=sys.stderr)

        # Start best-first search
        while not end:
            # Obtain hypothesis to expand
            empty, hyp = self.get_hypothesis_to_expand(priority_queue, stdict)
            # Check if priority queue is empty
            if empty:
                end = True
            else:
                # Expand hypothesis
                if verbose:
                    print("** niter:", niter, " ; lp:", hyp.score, ";", str(hyp.data), file=sys.stderr)
                # Stop if the hypothesis is complete
                if self.hyp_is_complete(hyp, src_word_array):
                    end = True
                else:
                    # Expand hypothesis
                    for l in range(0, config.a_par):
                        new_hyp_cov = self.last_cov_pos(hyp.data.coverage) + 1 + l
                        if new_hyp_cov < len(src_word_array):
                            # Obtain expansion
                            exp_list = self.expand(src_word_array, hyp, new_hyp_cov, verbose)
                            # Insert new hypotheses
                            for k in range(len(exp_list)):
                                # Insert hypothesis
                                priority_queue.put(exp_list[k])
                                # Update state info dictionary
                                sti = obtain_state_info(self.tmodel, self.lmodel, exp_list[k])
                                stdict.insert(sti, exp_list[k].score)

            niter += 1

            if niter > config.maxniters:
                end = True

        # Return result
        if niter > config.maxniters:
            if verbose:
                print("Warning: maximum number of iterations exceeded", file=sys.stderr)
            return Hypothesis()
        else:
            if self.hyp_is_complete(hyp, src_word_array):
                if verbose:
                    print(
                        "*** Best first search finished successfully after",
                        niter, "iterations, hyp. score:", hyp.score, file=sys.stderr
                    )
                hyp.score = hyp.score
                return hyp
            else:
                if verbose:
                    print(
                        "Warning: priority queue empty, search was unable to reach a complete hypothesis",
                        file=sys.stderr
                    )
                return Hypothesis()

    def detokenize(self, line, verbose=False):
        # Obtain array with tokenized words
        tok_array = split_string_to_words(line)
        nblsize = 1
        if verbose:
            print("**** Processing sentence: ", line.encode("utf-8"), file=sys.stderr)

        if len(tok_array) > 0:
            # Transform array of tokenized words
            trans_tok_array = []
            for i in range(len(tok_array)):
                trans_tok_array.append(transform_word(tok_array[i]))

            # Obtain n-best list of detokenized sentences
            nblist = self.obtain_nblist(trans_tok_array, nblsize, verbose)

            # Print detokenized sentence
            if len(nblist) == 0:
                print("Warning: no detokenizations were found for sentence in line", file=sys.stderr)
                return line
            else:
                best_hyp = nblist[0]
                detok_sent = self.obtain_detok_sent(tok_array, best_hyp)
                return detok_sent

        return ""

    def recase(self, line, verbose):
        lc_word_array = split_string_to_words(line)
        nblsize = 1
        if verbose:
            print("**** Processing sentence: ", line.encode("utf-8"), file=sys.stderr)
        if len(lc_word_array) > 0:
            # Obtain n-best list of detokenized sentences
            nblist = self.obtain_nblist(lc_word_array, nblsize, verbose)

            # Print recased sentence
            if len(nblist) == 0:
                print("Warning: no recased sentences were found for sentence:", line, file=sys.stderr)
                return line
            else:
                best_hyp = nblist[0]
                return best_hyp.data.words
        return ""

default_atoms = [
    "'em",
    "'ol",
    "vs.",
    "Ms.",
    "Mr.",
    "Dr.",
    "Mrs.",
    "Messrs.",
    "Gov.",
    "Gen.",
    "Mt.",
    "Corp.",
    "Inc.",
    "Co.",
    "co.",
    "Ltd.",
    "Bros.",
    "Rep.",
    "Sen.",
    "Jr.",
    "Rev.",
    "Adm.",
    "St.",
    "a.m.",
    "p.m.",
    "1a.m.",
    "2a.m.",
    "3a.m.",
    "4a.m.",
    "5a.m.",
    "6a.m.",
    "7a.m.",
    "8a.m.",
    "9a.m.",
    "10a.m.",
    "11a.m.",
    "12a.m.",
    "1am",
    "2am",
    "3am",
    "4am",
    "5am",
    "6am",
    "7am",
    "8am",
    "9am",
    "10am",
    "11am",
    "12am",
    "p.m.",
    "1p.m.",
    "2p.m.",
    "3p.m.",
    "4p.m.",
    "5p.m.",
    "6p.m.",
    "7p.m.",
    "8p.m.",
    "9p.m.",
    "10p.m.",
    "11p.m.",
    "12p.m.",
    "1pm",
    "2pm",
    "3pm",
    "4pm",
    "5pm",
    "6pm",
    "7pm",
    "8pm",
    "9pm",
    "10pm",
    "11pm",
    "12pm",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "May.",
    "Jun.",
    "Jul.",
    "Aug.",
    "Sep.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec.",
    "Ala.",
    "Ariz.",
    "Ark.",
    "Calif.",
    "Colo.",
    "Conn.",
    "Del.",
    "D.C.",
    "Fla.",
    "Ga.",
    "Ill.",
    "Ind.",
    "Kans.",
    "Kan.",
    "Ky.",
    "La.",
    "Md.",
    "Mass.",
    "Mich.",
    "Minn.",
    "Miss.",
    "Mo.",
    "Mont.",
    "Nebr.",
    "Neb.",
    "Nev.",
    "N.H.",
    "N.J.",
    "N.M.",
    "N.Y.",
    "N.C.",
    "N.D.",
    "Okla.",
    "Ore.",
    "Pa.",
    "Tenn.",
    "Va.",
    "Wash.",
    "Wis.",
    ":)",
    "<3",
    ";)",
    "(:",
    ":(",
    "-_-",
    "=)",
    ":/",
    ":>",
    ";-)",
    ":Y",
    ":P",
    ":-P",
    ":3",
    "=3",
    "xD",
    "^_^",
    "=]",
    "=D",
    "<333",
    ":))",
    ":0",
    "-__-",
    "xDD",
    "o_o",
    "o_O",
    "V_V",
    "=[[",
    "<33",
    ";p",
    ";D",
    ";-p",
    ";(",
    ":p",
    ":]",
    ":O",
    ":-/",
    ":-)",
    ":(((",
    ":((",
    ":')",
    "(^_^)",
    "(=",
    "o.O",
    "a.",
    "b.",
    "c.",
    "d.",
    "e.",
    "f.",
    "g.",
    "h.",
    "i.",
    "j.",
    "k.",
    "l.",
    "m.",
    "n.",
    "o.",
    "p.",
    "q.",
    "s.",
    "t.",
    "u.",
    "v.",
    "w.",
    "x.",
    "y.",
    "z.",
    "i.e.",
    "I.e.",
    "I.E.",
    "e.g.",
    "E.g.",
    "E.G."
]

_default_word_chars = \
    u"-.&" \
    u"0123456789" \
    u"ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
    u"abcdefghijklmnopqrstuvwxyz" \
    u"ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß" \
    u"àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ" \
    u"ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğ" \
    u"ĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁł" \
    u"ńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧ" \
    u"ŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ" \
    u"ΑΒΓΔΕΖΗΘΙΚΛΜΝΟΠΡΣΤΥΦΧΨΩΪΫ" \
    u"άέήίΰαβγδεζηθικλμνξοπρςστυφχψω"


class Tokenizer:
    def __init__(self, atoms=default_atoms, word_chars=_default_word_chars):
        """Initializer.
        Atom is a string that won't be split into separate tokens.  Longer
        atoms take precedence over their prefixes, e.g.: if 'a' and 'ab' are
        passed as atoms 'ab' will be returned.
        """

        atoms = list(map(re.escape, sorted(atoms, key=len, reverse=True)))
        word_chars = re.escape(word_chars)

        self.re = re.compile("(?:" + "|".join(
            atoms + [
                "\\b[0-9]+,[0-9]+[a-zA-Z]+\\b",
                "\\b[0-9]+,[0-9]+\\b",
                "[%s]+(?:'[sS])?" % word_chars,
                "\s+"
            ]) + ")", flags=re.I)

    def tokenize(self, text):
        """Tokenize text :: string -> [strings].
        Concatenation of returned tokens should yields.
        """

        # Following is a safeguard that guarantees that concatenating resultant
        # tokens yield original text.  It fills gaps between matches returned
        # by findall().  Ideally, it shouldn't be needed.  findall() should
        # return all required substrings.
        #
        # However, REs are tricky, mistakes happen, therefore we choose to be
        # defensive:
        matches = self.re.findall(text)
        tokens = []
        p = 0
        for match in matches:
            mp = text[p:].find(match)
            if mp != 0:
                missed = text[p:p + mp]
                tokens.append(missed)
            p = p + mp + len(match)
            tokens.append(match)

        if p < len(text):
            tokens.append(text[p:])
        return filter(lambda s: s.strip(), tokens)


def tokenize(string):
    tokenizer = Tokenizer()
    skel = list(annotated_string_to_xml_skeleton(string))
    for idx, (is_tag, txt) in enumerate(skel):
        if is_tag:
            skel[idx][1] = [skel[idx][1]]
        else:
            skel[idx][1] = tokenizer.tokenize(txt)
    return xml_skeleton_to_tokens(skel)


def xml_skeleton_to_tokens(skeleton):
    """
    Joins back the elements in a skeleton to return a list of tokens
    """
    annotated = []
    for _, tokens in skeleton:
        annotated.extend(tokens)
    return annotated


def lowercase(string):
    # return str.lower()
    skel = []
    for is_tag, txt in annotated_string_to_xml_skeleton(string):
        skel.append(
            (is_tag, txt.strip() if is_tag else txt.lower().strip())
        )

    return xml_skeleton_to_string(skel)


def xml_skeleton_to_string(skeleton):
    """
    Joins back the elements in a skeleton to return an annotated string
    """
    return u" ".join(txt for _, txt in skeleton)


def annotated_string_to_xml_skeleton(annotated):
    """
    Parses a string looking for XML annotations
    returns a vector where each element is a pair (is_tag, text)
    """
    offset = 0
    for m in config._annotation.finditer(annotated):
        if offset < m.start():
            yield [False, annotated[offset:m.start()]]
        offset = m.end()
        g = m.groups()
        dic_g = [x for x in g[0:8] if x]
        len_g = [x for x in g[8:11] if x]
        if dic_g:
            yield [True, dic_g[0]]
            yield [True, dic_g[1]]
            yield [False, dic_g[2]]
            yield [True, dic_g[3]]
            yield [True, dic_g[4]]
            yield [False, dic_g[5]]
            yield [True, dic_g[6]]
            yield [True, dic_g[7]]
        elif len_g:
            yield [True, len_g[0]]
            yield [False, len_g[1]]
            yield [True, len_g[2]]
        else:
            sys.stderr.write('WARNING:\n - s: %s\n - g: %s\n' % (annotated, g))
    if offset < len(annotated):
        yield [False, annotated[offset:]]


def remove_xml_annotations(annotated):
    xml_tags = {'<' + config.src_ann + '>', '</' + config.len_ann + '>', '</' + config.grp_ann + '>'}
    skeleton = list(annotated_string_to_xml_skeleton(annotated))
    tokens = []
    for i, is_tag, text in enumerate(skeleton):
        token = text.strip()
        if not is_tag and token:
            if i == 0:
                tokens.append(token)
            else:
                ant_is_tag, ant_text = skeleton[i - 1]
                if (
                    not ant_is_tag or
                    (ant_is_tag and ant_text.strip() in xml_tags)
                ):
                    tokens.append(token)
    return u' '.join(tokens)
