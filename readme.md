# Warmlab

## What is a WARM?

A WA-reinfrocement model (WARMs) is an interesting stochastic process inspired by the development of human brain \[1\].
Consider a graph $G$ with vertices $V$ and edges $E$.
Initially we start with some counts $N^{(0)}_e$ (typically $N^{(0)}_e=1$) for all the edges $e\in E$.
At each time step $t$, we select a vertex $v$ and reinforce an edge $e$ incident to $v$ randomly using their counts 
as weights, i.e., the probability of selecting $e$ is
$$\frac{N^{(t)}_e}{\sum_{f\text{ incident to }v} N^{(t)}_f}\,.$$

After the reinforcement, $N^{(t+1)}_e = N^{(t)}_e + 1$ for the chosen $e$ and the counts for other edges remain the same.
Interested readers can have a look at \[1\] for more details on the model and its interesting properties.

This package implements useful utilities for studying WA-reinforcement models, written as a part of my MSc thesis project in mathematics and statistics.


## Usage

### Building from Source and Installation

Clone the project and navigate to the root directory.
To install the package, build the project using
```python
python -m build .
```
It should be fairly quick and this generates a `dist` folder.
Locate `warm-vx.x.x.tar.gz` under that folder where the version is the current version and run
```python
pip install -r requirements.txt
pip install dist/warm-vx.x.x.tar.gz
```
to complete the installtion, after which you can `import` it as any other Python package.

### High-performance Computing Deployment

Doc to be written.

### Developing the Package

Clone the project to your local directory.
In order to avoid dependency clashes when working with the project, it is highly recommended to setup a virtual environment with
```python
virtualenv warmlab
```
and install the dependencies *only*
```python
pip install -r requirements.txt
```

Notice that we are not builidng the package nor installing the package itself, because we do not want older code to mess with the newer code in the environment.
To run the code under a development environment, use module-level run commands under the project root, e.g.
```python
python -m warmlab.warm
```

## References

[1] Remco van der Hofstad et al. “Strongly reinforced Pólya urns with graph-based
competition”. In: The Annals of Applied Probability 26.4 (2016), pp. 2494–2539.
