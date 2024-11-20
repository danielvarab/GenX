# Generative Extractive Summarization (GenX)

This repository contains the implementations for the algorithms presented in our paper "[Abstractive Summarizers are Excellent Extractive Summarizers](https://aclanthology.org/2023.acl-short.29/)" published at ACL 2023.


## Example Code

```python
import nltk

from inference import Cached


# Use the BRIO 
checkpoint = 'Yale-LILY/brio-cnndm-uncased'
genx = Cached.from_pretrained(checkpoint)

# Random news article
# https://www.cnn.com/2023/07/12/europe/italy-heat-wave-record-temperatures-climate-intl
text = open("data.txt").read()
sents = nltk.sent_tokenize(text)

output = genx(text, sents)

print(output["summary"])

# Output:
#
# A blistering and deadly heat wave in Italy this week could
# break records, with temperatures predicted to soar past 45
# degrees Celsius (113 Fahrenheit) in some parts of the
# country. The heat has already claimed at least one life.
# Italy’s Health ministry has issued a red alert (meaning
# “risk of death”) in 27 cities this week, including Rome,
# Florence and Bologna.
```

## Citation
```
@inproceedings{varab-xu-2023-abstractive,
    title = "Abstractive Summarizers are Excellent Extractive Summarizers",
    author = "Varab, Daniel  and
      Xu, Yumo",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.29",
    pages = "330--339",
    abstract = "Extractive and abstractive summarization designs have historically been fragmented, limiting the benefits that often arise from compatible model architectures. In this paper, we explore the potential synergies of modeling extractive summarization with an abstractive summarization system and propose three novel inference algorithms using the sequence-to-sequence architecture. We evaluate them on the CNN {\&} Dailymail dataset and show that recent advancements in abstractive system designs enable abstractive systems to not only compete, but even surpass the performance of extractive systems with custom architectures. To our surprise, abstractive systems achieve this without being exposed to extractive oracle summaries and, therefore, for the first time allow a single model to produce both abstractive and extractive summaries. This evidence questions our fundamental understanding of extractive system design, and the necessity for extractive labels while pathing the way for promising research directions in hybrid models.",
}
```
