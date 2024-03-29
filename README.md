# Virtual Analog Modeling Using Neural ODEs

This is the repository assiociated with the paper "Virtual Analog Modeling Of Distortion Circuits Using Neural
Ordinary Differential Equations" presented at the 25th International Conference on Digital Audio Effects (DAFx20in22) in Vienna, Austria, September 2022.

The publication is available on [arxiv](https://arxiv.org/abs/2205.01897) or on the [conference's website.](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_12.pdf)

Audio examples can be found at [the publication's examples page.](https://thewolfsound.com/publications/dafx2022/)

## Bibtex Citation

When you cite this work, please use the following bibliographical data:

```bibtex
@InProceedings{Wilczeketal2022,
  author    = {Wilczek, Jan and Wright, Alec and Välimäki, Vesa and Habets, Emanuël},
  booktitle = {Proceedings of the 25th International Conference on Digital Audio Effects (DAFx20in22), Vienna, Austria, September 2020-22},
  title     = {Virtual {A}nalog {M}odeling of {D}istortion {C}ircuits {U}sing {N}eural {O}rdinary {D}ifferential {E}quations},
  year      = {2022}
}
```

## Authors

* Jan Wilczek
* Alec Wright
* Vesa Välimäki
* Emanuël Habets

## Abstract

Recent research in deep learning has shown that neural networks
can learn differential equations governing dynamical systems. In
this paper, we adapt this concept to Virtual Analog (VA) modeling
to learn the ordinary differential equations (ODEs) governing the
first-order and the second-order diode clipper. The proposed models achieve performance comparable to state-of-the-art recurrent
neural networks (RNNs) albeit using fewer parameters. We show
that this approach does not require oversampling and allows to increase the sampling rate after the training has completed, which results in increased accuracy. Using a sophisticated numerical solver
allows to increase the accuracy at the cost of slower processing.
ODEs learned this way do not require closed forms but are still
physically interpretable.


## Contributions

* SPICE diode clipper model and its target signal provided by Alec Wright of Aalto University.
