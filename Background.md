Why deep linear models tend to “go low-rank”

Consider supervised linear regression with a factorized parameterization:
\hat y = (WU)x,\quad W\in\mathbb R^{m\times r},\ U\in\mathbb R^{r\times d},
trained by gradient descent on squared loss over data (x_i,y_i). Let A:=WU be the end-to-end linear map.

If you trained A directly (no factorization), gradient descent from small initialization typically converges to the minimum-Frobenius-norm solution that fits the data (in the separable/noiseless regime). But with the factorization A=WU, the optimization lives in a larger space, and gradient descent has an implicit bias toward low nuclear norm / low effective rank solutions.

There are a few complementary ways to see it:

1) The factorization makes “spreading mass across many singular directions” expensive
Among all factorizations A=WU, there is a classic inequality:
\|A\|_* \;\le\; \tfrac12(\|W\|_F^2+\|U\|_F^2),
and equality can be achieved when the factors are balanced (roughly, they share the singular values evenly). So, controlling the Frobenius norms of W and U is (in effect) controlling the nuclear norm of A, which is the convex surrogate for rank.

Gradient descent from small random init tends to keep \|W\|_F,\|U\|_F relatively small while fitting the data, which pushes it toward small \|A\|_*—and that typically means few large singular values (low effective rank).

2) Singular modes evolve like “rich get richer”
In the simplest setting (full-batch GD, whitened inputs so the loss behaves like \tfrac12\|A-A_*\|_F^2), the gradient flow dynamics are
\dot W = -(A-A_*)U^\top,\qquad \dot U = -W^\top(A-A_*).
When you rotate into the singular vector basis of the target A_*, the dynamics approximately decouple by singular mode. For a mode i with end-to-end singular value s_i(t), you get a growth law of the flavor
\dot s_i \propto s_i(\sigma_i - s_i)
(up to details and “balancing” assumptions), where \sigma_i is the target singular value. This logistic-ish form implies:
	•	If s_i is tiny, it grows slowly.
	•	Once it’s nontrivial, it grows faster.
So a small subset of modes “turn on” early and dominate, while many modes remain near zero for a long time → low effective rank during training (and often in the limit too, depending on data/noise/early stopping).

3) Rank is not explicitly penalized, but the parameterization creates an implicit regularizer
Because A=WU is nonconvex, there are many global minima (many factorizations) that fit the training data. Gradient descent doesn’t pick one arbitrarily: with small init it tends to find solutions with
	•	balanced factors (no giant scale mismatch between W and U),
	•	smaller \|W\|_F^2+\|U\|_F^2,
which corresponds closely to smaller nuclear norm of A, hence low rank / low effective rank.
