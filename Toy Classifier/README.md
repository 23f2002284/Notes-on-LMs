# Inspired by 
Approximating Attention Article: [Approximating Attention](https://sea-snell.github.io/AttentionBlogSite/attention/2021/04/01/attention.html)

## A toy classifier
a toy Classification task
# Task
A Keyword detection classification task     
each sequence consists of either 0 or 1 as token

Classification label is +ve if  "1" appears in sequence.
ex:
{0, 0} -> Negative
{0, 1} -> Positive
# Classifier
Here we are building an Unigram gated-Classifier.
It Learns a unique weight for each vocab item. Vocab 0 is associated with a classifier weight $W_0$ and Vocab 1 is associated with a Classifier weight $W_1$ .
These unigram $W$ weights are gated by a learned "Attention weight" $P$. which is 0 < $P$ <1, corresponding to how much the classifier should attend to vocab 1 and as opposite to 0 vocab. (attention weight as ($1-P$)).
and 
the learned attention weight $P$ only applied if both vocab symbols appear in an input sequence,
otherwise all the attention (attention weight = 1) is placed on the one symbol in the sequence. ( as it is in the blog )

$s_0 = c({0, 0})=W_0$
$s_1= c({0, 1})=(1-P) W_0 + P W_1$
So it is mentioned that we should get a positive weight $W_1$ as it is associated with positive and the same way for $W_0$ to be negative. 
we want the classifier to learn to attend the key token, vocab 1 with P converging to 1.

# Gradient
gradient descent with very small step size. 
the parameters $\beta_0 \ , \beta_1\ , P$  
Let the Training time be $\tau$ 
so the changes of parameters over the training time is as follows
$$
\begin{align}
\frac{\partial W_{0}(\tau)} {\partial \tau} &= -\frac{\partial \mathcal{L}}{\partial W_{0}}=P \\ \\
\frac{\partial W_{1}(\tau)} {\partial \tau} &= -\frac{\partial \mathcal{L}}{\partial W_{1}}=P \\
\frac{\partial P(\tau)}{\partial \tau} &= \frac{\partial \mathcal{L}}{\partial \tau} = W_{1}-W_{0}
\end{align}
$$
where P is unbounded here.
# Dynamics
1. How this will evolve with time ?
let's say $p>0$, then  it simply mean weight for 1 token $W_{1}$ will increase every time and $W_0$ will decrease every time.

as $W_{1}$ is associated with positive sequence and $W_{0}$ with negative sequence. $P$ will not necessarily increase towards 1 as it depends on the value with $W_{1}-W_{0}$.

if uniformly initialized classifier i.e. say $W_{0}=W_{1},P=0.5$ then initially $P$ will have 0 gradient and remain at 0.5, while  $W_{0}$ will decrease and $W_{1}$ will increase later $W_{1}-W_{0}>0$ will continue to increase, causing $P$ to also increase and converge to 1.

Simply
the Toy Classifer first to learn that vocab = 0 is associated with negative label and vocab 1 with postive label unders uniform attention and then it learns $P$
