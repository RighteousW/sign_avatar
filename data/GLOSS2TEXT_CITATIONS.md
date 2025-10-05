# Citations and References

## Primary Research Papers

This implementation is based on research from two key papers:

### 1. Sequence-to-Sequence Attention Models (2019)

**Paper**: "Translation of Sign Language Glosses to Text Using Sequence-to-Sequence Attention Models"

**Authors**: Nikolaos Arvanitis, Constantinos Constantinopoulos, Dimitrios Kosmopoulos

**Published**: 2019 15th International Conference on Signal-Image Technology & Internet-Based Systems (SITIS)

**DOI**: 10.1109/SITIS.2019.00056

**Key Contributions Used**:
- GRU-based encoder-decoder architecture with attention mechanism
- Three attention functions: dot, general, and concat (Luong attention)
- Teacher forcing with 0.5 probability
- Dropout rate of 0.25
- Adamax optimizer with learning rate 0.001
- Architecture configurations: 2-layer (350 hidden units) and 4-layer (800 hidden units)
- BLEU score evaluation methodology
- Use of ASLG-PC12 dataset

**Citation**:
```
N. Arvanitis, C. Constantinopoulos and D. Kosmopoulos, 
"Translation of Sign Language Glosses to Text Using Sequence-to-Sequence Attention Models," 
2019 15th International Conference on Signal-Image Technology & Internet-Based Systems (SITIS), 
Sorrento, Italy, 2019, pp. 296-302, doi: 10.1109/SITIS.2019.00056.
```

### 2. Non-Autoregressive Models (2025)

**Paper**: "Non-autoregressive Modeling for Sign-gloss to Texts Translation"

**Authors**: Fan Zhou, Tim Van de Cruys

**Published**: Proceedings of Machine Translation Summit XX, Volume 1, pages 220–230, June 23-27, 2025

**Affiliation**: Center for Computational Linguistics, KU Leuven

**Key Contributions Referenced**:
- Comparison between autoregressive and non-autoregressive approaches
- Performance benchmarks on ASLG-PC12 dataset
- Inference speed measurements
- Understanding of gloss-to-text translation challenges

**Citation**:
```
Fan Zhou and Tim Van de Cruys. 2025. 
Non-autoregressive Modeling for Sign-gloss to Texts Translation. 
In Proceedings of Machine Translation Summit XX Volume 1, pages 220–230.
```

## Dataset

**ASLG-PC12 (American Sign Language Gloss - Parallel Corpus 2012)**

**Authors**: Achraf Othman, Mohamed Jemni

**Published**: 5th Workshop on the Representation and Processing of Sign Languages LREC12, May 2012

**Description**: Parallel corpus containing approximately 87,710 gloss-text pairs for American Sign Language

**Citation**:
```
Achraf Othman and Mohamed Jemni. 2012. 
English-ASL Gloss Parallel Corpus 2012: ASLG-PC12. 
In Proceedings of the 5th Workshop on the Representation and Processing of Sign Languages: 
Interactions between Corpus and Lexicon LREC.
```

## Attention Mechanisms

**Luong Attention**

**Paper**: "Effective Approaches to Attention-based Neural Machine Translation"

**Authors**: Minh-Thang Luong, Hieu Pham, Christopher Manning

**Published**: Conference on Empirical Methods in Natural Language Processing (EMNLP) 2015

**Citation**:
```
Minh-Thang Luong, Hieu Pham, and Christopher Manning. 2015. 
Effective Approaches to Attention-based Neural Machine Translation. 
In Conference on Empirical Methods in Natural Language Processing (EMNLP) 2015.
```

**Bahdanau Attention**

**Paper**: "Neural Machine Translation by Jointly Learning to Align and Translate"

**Authors**: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio

**Published**: arXiv:1409.0473v7, 2014

**Citation**:
```
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. 
Neural Machine Translation by Jointly Learning to Align and Translate. 
arXiv:1409.0473v7
```

## Evaluation Metrics

**BLEU Score**

**Paper**: "BLEU: a method for automatic evaluation of machine translation"

**Authors**: Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu

**Published**: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL '02), 2002

**Citation**:
```
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. 
BLEU: a method for automatic evaluation of machine translation. 
In Proceedings of the 40th Annual Meeting on Association for Computational Linguistics (ACL '02). 
Association for Computational Linguistics, Stroudsburg, PA, USA, 311-318. 
DOI: https://doi.org/10.3115/1073083.1073135
```

## Neural Network Architectures

**GRU (Gated Recurrent Unit)**

**Paper**: "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"

**Authors**: Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio

**Published**: 2014

**Citation**:
```
Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. 2014. 
Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
```

**Sequence-to-Sequence Learning**

**Paper**: "Sequence to Sequence Learning with Neural Networks"

**Authors**: Ilya Sutskever, Oriol Vinyals, Quoc V. Le

**Published**: Advances in Neural Information Processing Systems (NIPS 2014)

**Citation**:
```
Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. 
Sequence to sequence learning with neural networks. 
In Advances in Neural Information Processing Systems (NIPS 2014).
```

## Optimization Algorithms

**Adam/Adamax Optimizer**

**Paper**: "Adam: A Method for Stochastic Optimization"

**Authors**: Diederik P. Kingma, Jimmy Lei Ba

**Published**: ICLR 2015

**Citation**:
```
Diederik P. Kingma and Jimmy Lei Ba. 2015. 
Adam: A method for stochastic optimization. 
In ICLR 2015, Ithaca.
```

## Software and Tools

**PyTorch**: https://pytorch.org/

**NLTK (Natural Language Toolkit)**: https://www.nltk.org/
```
Edward Loper and Steven Bird. 2002. 
NLTK: the Natural Language Toolkit. 
In Proceedings of the ACL-02 Workshop on Effective tools and methodologies 
for teaching natural language processing and computational linguistics.
```

---

## Implementation Notes

This implementation combines methodologies from multiple research papers:

1. **Architecture and Training**: Based primarily on Arvanitis et al. (2019), using their GRU encoder-decoder with Luong attention, hyperparameters, and training procedures.

2. **Performance Benchmarks**: Referenced from Zhou & Van de Cruys (2025) for understanding autoregressive vs non-autoregressive trade-offs.

3. **Dataset**: Uses the ASLG-PC12 corpus format from Othman & Jemni (2012).

4. **Evaluation**: BLEU score implementation following Papineni et al. (2002).

The code is an educational implementation for research and development purposes. Users should cite the relevant papers when using this implementation in academic work.

---

## License and Usage

When using this implementation, please cite:

1. The primary paper you're following (Arvanitis et al., 2019 for this implementation)
2. The ASLG-PC12 dataset (Othman & Jemni, 2012) if using that data
3. Any additional papers whose methods you specifically utilize

**Example Citation in Your Paper**:
```
We implemented a GRU-based sequence-to-sequence model with Luong attention 
following the approach of Arvanitis et al. [1], trained on the ASLG-PC12 
dataset [2], and evaluated using BLEU scores [3].

[1] N. Arvanitis, C. Constantinopoulos and D. Kosmopoulos, 
    "Translation of Sign Language Glosses to Text Using Sequence-to-Sequence 
    Attention Models," 2019 SITIS Conference.
[2] A. Othman and M. Jemni, "English-ASL Gloss Parallel Corpus 2012," 
    LREC 2012.
[3] K. Papineni et al., "BLEU: a method for automatic evaluation of 
    machine translation," ACL 2002.
```