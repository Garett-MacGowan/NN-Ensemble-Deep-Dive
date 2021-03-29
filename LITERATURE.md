# Related Literature

---

## Deep Ensembles: A Loss Landscape Perspective
### Link
https://arxiv.org/pdf/1912.02757.pdf

### Authors
Stanislav Fort, Huiyi Hu, Balaji Lakshminarayanan

### Video Resources
Amazing video explanation by YouTube channel "Yannic Kilcher":

https://www.youtube.com/watch?v=5IRlUVrEVL8

### Garett's Abstract

Different local optima exist in the loss landscape of neural networks. For a given neural network architecture,
the different local optima in the loss landscape tend to achieve similar accuracy to one another, but can often be
functionally different. This functional difference is exemplified by the fact that, for a set of local optima with
similar loss and accuracy values, the samples which are correctly or incorrectly classified in the set vary greatly
across the different optima. This research provides reasoning to use an ensemble of neural networks, where each network
in the ensemble is initialized from scratch and trained to convergence.

Why are there different local optima? Here are some possibilities...
- Correlation != causation. Some correlations between variables seem to explain the outcomes, but in reality, 
  they don't, they are just correlated. 
- A problem could be "easy", and there are multiple ways to solve the same problem.
- There are symmetries in the model architecture, e.g. it is possible to swap two neurons and get the same result.


## Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs
### Link
https://arxiv.org/pdf/1802.10026.pdf

### Authors
Timur Garipov, Pavel Izmailov, Dimitrii Podoprikhin, Dmitry Vetrov, Andrew Gordon Wilson

### Video Resources:
Cool video containing ideas in this paper by YouTube channel "Seattle Applied Deep Learning":

https://www.youtube.com/watch?v=Z_MA8CWKxFU

### Paper's Abstract:
The loss functions of deep neural networks are complex and their geometric
properties are not well understood. We show that the optima of these complex
loss functions are in fact connected by simple curves over which training and test
accuracy are nearly constant. We introduce a training procedure to discover these
high-accuracy pathways between modes. Inspired by this new geometric insight,
we also propose a new ensembling method entitled Fast Geometric Ensembling
(FGE). Using FGE we can train high-performing ensembles in the time required to
train a single model. We achieve improved performance compared to the recent
state-of-the-art Snapshot Ensembles, on CIFAR-10, CIFAR-100, and ImageNet.


## Averaging Weights Leads to Wider Optima and Better Generalization
### Link
https://arxiv.org/pdf/1803.05407.pdf

### Video Resources:
Presenting the paper:

https://www.youtube.com/watch?v=cbVZDFs46mg

### Authors
Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson

### Paper's Abstract:
Deep neural networks are typically trained by optimizing a loss function with an SGD variant, in conjunction with a 
decaying learning rate, until convergence. We show that simple averaging of multiple points along the trajectory of SGD,
with a cyclical or constant learning rate, leads to better generalization than conventional training. We also show that
this Stochastic Weight Averaging (SWA) procedure finds much flatter solutions than SGD, and approximates the recent
Fast Geometric Ensembling (FGE) approach with a single model. Using SWA we achieve notable improvement in test accuracy
over conventional SGD training on a range of state-of-the-art residual networks, PyramidNets, DenseNets, and 
Shake-Shake networks on CIFAR-10, CIFAR-100, and ImageNet. In short, SWA is extremely easy to implement, improves 
generalization, and has almost no computational overhead.


## On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima
### Link
https://arxiv.org/abs/1609.04836

### Authors
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang

### Paper's Abstract:
The stochastic gradient descent (SGD) method and its variants are algorithms of choice for many Deep Learning tasks. 
These methods operate in a small-batch regime wherein a fraction of the training data, say 32-512 data points, 
is sampled to compute an approximation to the gradient. It has been observed in practice that when using a larger batch
there is a degradation in the quality of the model, as measured by its ability to generalize. We investigate the cause
for this generalization drop in the large-batch regime and present numerical evidence that supports the view that
large-batch methods tend to converge to sharp minimizers of the training and testing functions - and as is well known,
sharp minima lead to poorer generalization. In contrast, small-batch methods consistently converge to flat minimizers,
and our experiments support a commonly held view that this is due to the inherent noise in the gradient estimation.
We discuss several strategies to attempt to help large-batch methods eliminate this generalization gap.


## Visualizing the Loss Landscape of Neural Nets
### Link
https://proceedings.neurips.cc/paper/2018/file/a41b3bb3e6b050b6c9067c67f663b915-Paper.pdf

### Authors
Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein

### Paper's Abstract:
Neural network training relies on our ability to find “good” minimizers of highly
non-convex loss functions. It is well-known that certain network architecture
designs (e.g., skip connections) produce loss functions that train easier, and well chosen 
training parameters (batch size, learning rate, optimizer) produce minimizers 
that generalize better. However, the reasons for these differences, and their
effect on the underlying loss landscape, are not well understood. In this paper, we
explore the structure of neural loss functions, and the effect of loss landscapes on
generalization, using a range of visualization methods. First, we introduce a simple
“filter normalization” method that helps us visualize loss function curvature and
make meaningful side-by-side comparisons between loss functions. Then, using
a variety of visualizations, we explore how network architecture affects the loss
landscape, and how training parameters affect the shape of minimizers.