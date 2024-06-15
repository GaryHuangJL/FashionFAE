
import re
from nltk.corpus import wordnet as wn
import random
import copy

def process_description(sent, tagger=None, tokenizer=None):

    sent_tokens = tokenizer.tokenize(pre_caption(sent))
    original_tokens = tokenizer.tokenize(pre_caption(sent))
    tokenizer_vocab = list(tokenizer.vocab.keys())

    tokens_length = len(sent_tokens)
    mask_token = []
    mask_sign = [0]*tokens_length
    replace_token = []
    replace_sign = [0]*tokens_length

    pos_index = list(range(tokens_length))
    random.shuffle(pos_index)
    tokenizer_vocab = ['[MASK]'] if tokenizer_vocab is None else tokenizer_vocab

    #### mask sentence
    mask_length = int(tokens_length * 0.15 * 0.8)
    for i in range(mask_length):
        mask_token.append(sent_tokens[pos_index[i]])
        mask_sign[pos_index[i]] = 1
        sent_tokens[pos_index[i]] = '[MASK]'
        replace_length = int(tokens_length * 0.15 * 0.1)
    
    
    replace_length = int(tokens_length * 0.15 * 0.1)    
    for i in range(mask_length, mask_length+replace_length):
        mask_token.append(sent_tokens[pos_index[i]])
        mask_sign[pos_index[i]] = 1

        replace_token.append(sent_tokens[pos_index[i]])
        replace_sign[pos_index[i]] = 1
        if random.random() < 0.5:
            sent_tokens[pos_index[i]] = random.choice(tokenizer_vocab)
        else:
            sent_tokens[pos_index[i]] = search_antonym(sent_tokens[pos_index[i]])


    added_tokens = tokenizer.tokenize('the image description is')
    original_tokens = added_tokens + original_tokens
    sent_tokens = added_tokens + sent_tokens
    replace_sign = [0]* (len(added_tokens)) + replace_sign
    mask_sign = [0] * (len(added_tokens)) + mask_sign


    return original_tokens,sent_tokens, mask_token, mask_sign ,replace_token, replace_sign




def process_attribute(attribute_dict, tokenizer) -> list:
    outstrs = []
    val_tokenizer = []
    titlevalue = []
    prompt_tokens = tokenizer.tokenize('the image ')
    attribute_names = list(attribute_dict.keys())

    attribute_names_wotitle = attribute_names[1:]
    random.shuffle(attribute_names_wotitle)

    name = attribute_names[0]
    val = attribute_dict[name]
    val_tokens = val.split(' ')
    name = tokenizer.tokenize(name)
    for v in val_tokens:
        val_tokenizer.append(tokenizer.tokenize(v))
    orititlevalue = []
    for v in val_tokenizer:
        for k in v:
            orititlevalue.append(k)

    a = len(val_tokenizer)

    listtitle = [0] * (a - 2) + [1] * 2
    random.shuffle(listtitle)
    mask_token = []
    for i in range(a):
        if listtitle[i] == 1:
            mask_token = mask_token + val_tokenizer[i]
            val_tokenizer[i] = ['[MASK]'] * len(val_tokenizer[i])

    for v in val_tokenizer:
        for k in v:
            titlevalue.append(k)
    
    out = prompt_tokens + name + ['is'] + titlevalue
    oriout = prompt_tokens + name + ['is'] + orititlevalue
    
    for name in attribute_names_wotitle:
        outstrs.append((name, attribute_dict[name]))

    ##### process
    for i, v in enumerate(outstrs):
        if v[0] == 'subcategory':
            outstrs[i] = ('sub category', v[1])

    out_tokens = []

    for attr in outstrs:
        name, val = attr
        out_tokens.append(tokenizer.tokenize(name.lower()))
        out_tokens.append(tokenizer.tokenize(val.replace('&', 'and').lower()))

    ori_tokens = copy.deepcopy(out_tokens)

    prompt_val_flag = 1
    mask_token = mask_token + out_tokens[4 + prompt_val_flag]  # 4 because skip first and second (name-val) tuple
    out_tokens[4 + prompt_val_flag] = ['[MASK]'] * len(out_tokens[4 + prompt_val_flag])
    #mask_token = mask_token + out_tokens[prompt_val_flag]  # 4 because skip first and second (name-val) tuple
    #out_tokens[prompt_val_flag] = ['[MASK]'] * len(out_tokens[prompt_val_flag])

    for i in range(0, len(out_tokens), 2):
        out_tokens[i] = prompt_tokens + out_tokens[i] + ['is']

    for t in range(0, len(ori_tokens), 2):
        ori_tokens[t] = prompt_tokens + ori_tokens[t] + ['is']
  
    out_tokens = [i for k in out_tokens for i in k]
    ori_tokens = [j for v in ori_tokens for j in v]
    out_tokens = out + out_tokens
    ori_tokens = oriout + ori_tokens
    
    #out_tokens = out
    #ori_tokens = oriout
    mask_sign = [0] * len(out_tokens)
    for i, token in enumerate(out_tokens):
        if token != '[MASK]':
            continue
        else:
            mask_sign[i] = 1

    replace_token = []
    replace_sign = [0] * len(out_tokens)

    return ori_tokens, out_tokens, mask_token, mask_sign, replace_token, replace_sign


def search_synonym(word, label=None):
    '''
    if finded return syn else return word
    '''
    assert label in ('n','a','v',None)
    syns = wn.synsets(word)
    syns_set = []
    for syn in syns:
        syns_set.extend(syn.lemma_names())
    syns_set = set(syns_set)
    if syns_set:
        word = random.choice(list(syns_set))

    return word


def search_antonym(word, label=None):
    anto = []
    for syn in wn.synsets(word):
        for lm in syn.lemmas():
            if lm.antonyms():
                anto.append(lm.antonyms()[0].name())
    return random.choice(anto) if anto else word


def pre_caption(caption,max_words=None):
    caption = re.sub(
        # r"([,.'!?\"()*#:;~])",
        r"([,.'!\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person').replace('<br>',' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    #truncate caption
    if max_words is not None:
        caption_words = caption.split(' ')
        if len(caption_words)>max_words:
            caption = ' '.join(caption_words[:max_words])

    return caption

def pharse_fashiongen_season(input_season):
    '''
    pharse season abbreviation to origin string tuble for fashiongen
    '''
    assert len(input_season) == 6
    season = input_season[:2]
    year = input_season[2:]
    assert season in ('SS','FW')
    if season == 'SS':
        season = 'spring summer'
    elif season == 'FW':
        season = 'fall winter'
    return season + ' ' + year
