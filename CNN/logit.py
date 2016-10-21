import numpy
import theano
import theano.tensor as T


class LinearRegression(object):
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        p_y_given_x = T.dot(input, self.W) + self.b
        self.y_pred = p_y_given_x[:, 0]

        # parameters of the model
        self.params = [self.W, self.b]

    def cost(self, y):
        # cost = -T.sqr(T.mean(T.pow(self.y_pred - y, 2)))
        # cost = T.sum(T.pow(prediction-y,2))
        # cost = T.sum(T.sqr(y - self.y_pred))
        # cost = T.sum(T.sqr(y - self.y_pred))
        # cost = T.mean(T.sqr(y - self.y_pred))
        # cost = T.sum(T.sqr((y - self.y_pred) ** 2))
        # cost = T.mean(T.sqr(T.abs_(y - self.y_pred)))
        # cost = T.mean((y - self.y_pred) ** 2)
        cost = T.mean(T.abs_(y - self.y_pred))
        return cost

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            # error = T.mean(T.neq(self.y_pred, y))
            # error = T.sqr(T.mean(T.pow(self.y_pred - y, 2)))
            # error = T.sum(T.sqr(y - self.y_pred))
            # error = T.mean(T.sqr(y - self.y_pred))
            # error = T.sum(T.sqr((y - self.y_pred) ** 2))
            error = T.sum(T.sqr(T.abs_(y - self.y_pred)))
            return error
        else:
            raise NotImplementedError()


class MultiLinearRegression(object):
    """Multi-class Linear Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting
    data points onto a set of hyperplanes, the distance to which is used
    to determine a class membership probability.
    """

    def __init__(self, input, n_in, n_outs):
        """ Initialize the parameters of the logistic regression

        :type n_outs: list of int
        :param n_outs: number of output units in each group

        """

        n_out = numpy.sum(n_outs)
        n_groups = len(n_outs)
        self.n_groups = n_groups

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX),
            name='W',
            borrow=True)

        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX),
            name='b',
            borrow=True)

        self.h = T.dot(input, self.W) + self.b
        self.p_y_given_x = []
        self.y_pred = []
        t = 0
        for idx in range(n_groups):
            p_y_given_x = self.h[:, t:t + n_outs[idx]]
            y_pred = p_y_given_x[:, 0]
            t += n_outs[idx]
            self.p_y_given_x.append(p_y_given_x)
            self.y_pred.append(y_pred)

            # parameters of the model
        self.params = [self.W, self.b]

    def cost(self, ys):
        cost = 0
        for idx in range(0, self.n_groups):
            cost += T.mean(T.sqr(ys[:, idx] - self.y_pred[idx]))
        return cost

    def errors(self, ys, n_classes):
        errs = []
        for idx in range(self.n_groups):
            if ys[:, idx].ndim != self.y_pred[idx].ndim:
                raise TypeError('y should have the same shape as self.y_pred',
                                ('y', ys[:, idx].type, 'y_pred', self.y_pred[idx].type))
                # check if y is of the correct datatype
            if ys[:, idx].dtype.startswith('int'):
                # error = T.mean(T.neq(self.y_pred[idx], ys[:, idx]))
                error = T.mean(T.sqr(self.y_pred[idx] - ys[:, idx]))
                errs.append(error)
            else:
                raise NotImplementedError()
        return errs


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            error = T.mean(T.neq(self.y_pred, y))
            return error
        else:
            raise NotImplementedError()


class DummyRegressionLayer(object):
    def __init__(self, input=None, target=None, regularize=True):
        super(RegressionLayer, self).__init__()  # boilerplate
        # MODEL CONFIGURATION
        self.regularize = regularize
        # ACQUIRE/MAKE INPUT AND TARGET
        if not input:
            input = T.matrix('input')
        if not target:
            target = T.matrix('target')
        # HYPER-PARAMETERS
        self.stepsize = T.scalar()  # a stepsize for gradient descent
        # PARAMETERS
        self.w = T.matrix()  # the linear transform to apply to our input points
        self.b = T.vector()  # a vector of biases, which make our transform affine instead of linear
        # REGRESSION MODEL
        self.activation = T.dot(input, self.w) + self.b
        self.prediction = self.build_prediction()
        # CLASSIFICATION COST
        self.classification_cost = self.build_classification_cost(target)
        # REGULARIZATION COST
        self.regularization = self.build_regularization()
        # TOTAL COST
        self.cost = self.classification_cost
        if self.regularize:
            self.cost = self.cost + self.regularization
        # GET THE GRADIENTS NECESSARY TO FIT OUR PARAMETERS
        self.grad_w, self.grad_b, grad_act = T.grad(self.cost, [self.w, self.b, self.prediction])

    def params(self):
        return self.w, self.b

    def _instance_initialize(self, obj, input_size=None, target_size=None,
                             seed=1827, **init):
        # obj is an "instance" of this module holding values for each member and
        # functions for each method
        if input_size and target_size:
            # initialize w and b in a special way using input_size and target_size
            sz = (input_size, target_size)
            rng = N.random.RandomState(seed)
            obj.w = rng.uniform(size=sz, low=-0.5, high=0.5)
            obj.b = N.zeros(target_size)
            obj.stepsize = 0.01
        # here we call the default_initialize method, which takes all the name: value
        # pairs in init and sets the property with that name to the provided value
        # this covers setting stepsize, l2_coef; w and b can be set that way too
        # we call it after as we want the parameter to superseed the default value.
        M.default_initialize(obj, **init)

    def build_regularization(self):
        return T.zero()  # no regularization!


class SoftMaxRegression(object):
    """Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = self.p_y_given_x.flatten(2)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            diff = y - self.y_pred
            diff, order = theano.scan(lambda x_i: T.sqrt((x_i ** 2).sum()), sequences=[diff])
            error = T.mean(diff)
            return diff
        else:
            raise NotImplementedError()


class MultiLogisticRegression(object):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting
    data points onto a set of hyperplanes, the distance to which is used
    to determine a class membership probability.
    """

    def __init__(self, input, n_in, n_outs):
        """ Initialize the parameters of the logistic regression

        :type n_outs: list of int
        :param n_outs: number of output units in each group

        """

        n_out = numpy.sum(n_outs)
        n_groups = len(n_outs)
        self.n_groups = n_groups

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                               name='W')
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b')

        self.h = T.dot(input, self.W) + self.b
        self.p_y_given_x = []
        self.y_pred = []
        t = 0
        for idx in range(n_groups):
            p_y_given_x = T.nnet.softmax(self.h[:, t:t + n_outs[idx]])
            y_pred = T.argmax(p_y_given_x, axis=1)
            t += n_outs[idx]
            self.p_y_given_x.append(p_y_given_x)
            self.y_pred.append(y_pred)

            # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, ys):
        """Return the mean of the negative log-likelihood of the
        prediction of this model under a given target distribution.

        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
                the correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities
        # (call it LP) with one row per example and one column per class
        # LP[T.arange(y.shape[0]),y] is a vector v containing
        # [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]]
        # and T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across
        # inibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        cost = 0
        for idx in range(0, self.n_groups):
            cost += -T.mean(T.log(self.p_y_given_x[idx])[T.arange(ys[:, idx].shape[0]), ys[:, idx]])
        return cost

    def errors(self, ys, n_classes):
        errs = []
        for idx in range(self.n_groups):
            if ys[:, idx].ndim != self.y_pred[idx].ndim:
                raise TypeError('y should have the same shape as self.y_pred',
                                ('y', ys[:, idx].type, 'y_pred', self.y_pred[idx].type))
                # check if y is of the correct datatype
            if ys[:, idx].dtype.startswith('int'):
                # the T.neq operator returns a vector of 0s and 1s, where 1
                # represents a mistake in prediction
                # the error should not be if classified correct or wrong
                # the error should be how close the predicted to the truth
                # in other words, we will draw the predicted region and the original region
                # and see how much is the difference
                error = T.mean(T.neq(self.y_pred[idx], ys[:, idx]))
                # error = T.mean(T.abs_(self.y_pred[idx] - ys[:, idx])) / n_classes
                errs.append(error)
            else:
                raise NotImplementedError()
        return errs

    def errors_trial(self, ys, n_classes, idx_start, idx_end):
        errs = []

        for ys_i in ys[idx_start:idx_end]:
            x = 7

        # for idx in range(idx_start, idx_end):
        #     if ys[idx, 0].ndim != self.y_pred[0][idx].ndim:
        #         raise TypeError('y should have the same shape as self.y_pred',
        #                         ('y', ys[idx, 0].type, 'y_pred', self.y_pred[0][idx].type))
        #     if ys[idx, 0].dtype.startswith('int'):
        #         #region_y = ys[idx, 0:4]
        #         #region_pred = [self.y_pred[0][idx], self.y_pred[1][idx], self.y_pred[2][idx], self.y_pred[3][idx]]
        #         # error = T.mean(T.abs_(self.y_pred[0][idx] - ys[0][idx])) / n_classes
        #         error = T.mean(ys[idx, 0])
        #         errs.append(error)
        #     else:
        #         raise NotImplementedError()
        #     idx += 1
        # return errs

        return 0

        # for idx in range(self.n_groups):
        #     if ys[:, idx].ndim != self.y_pred[idx].ndim:
        #         raise TypeError('y should have the same shape as self.y_pred',
        #                         ('y', ys[:, idx].type, 'y_pred', self.y_pred[idx].type))
        #         # check if y is of the correct datatype
        #     if ys[:, idx].dtype.startswith('int'):
        #         # the T.neq operator returns a vector of 0s and 1s, where 1
        #         # represents a mistake in prediction
        #         # the error should not be if classified correct or wrong
        #         # the error should be how close the predicted to the truth
        #         # in other words, we will draw the predicted region and the original region
        #         # and see how much is the difference
        #         # error = T.mean(T.neq(self.y_pred[idx], ys[:, idx]))
        #         error = T.mean(T.abs_(self.y_pred[idx] - ys[:, idx])) / n_classes
        #         errs.append(error)
        #     else:
        #         raise NotImplementedError()
        # return errs


def logit_layer(input, W, b):
    p_y_given_x = T.nnet.softmax(T.dot(input, W) + b)
    y_pred = T.argmax(p_y_given_x, axis=1)
    return y_pred, p_y_given_x


def multi_logit_layer(input, W, b, n_outs):
    n_groups = len(n_outs)

    h = T.dot(input, W) + b
    p_y_given_x = []
    y_pred = []
    t = 0
    for idx in range(n_groups):
        _p_y_given_x = T.nnet.softmax(h[:, t:t + n_outs[idx]])
        _y_pred = T.argmax(_p_y_given_x, axis=1)
        t += n_outs[idx]
        p_y_given_x.append(_p_y_given_x)
        y_pred.append(_y_pred)

    return y_pred, p_y_given_x


def classify_images(input_flatten, hidden_output, filters, W, b):
    """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

    # symbolic expression for computing the matrix of class-membership
    # probabilities
    # Where:
    # W is a matrix where column-k represent the separation hyper plain for
    # class-k
    # x is a matrix where row-j  represents input training sample-j
    # b is a vector where element-k represent the free parameter of hyper
    # plain-k

    s = filters.shape
    filters_reshaped = filters.reshape((s[0], s[1] * s[2] * s[3]))

    y_pred, p_y_given_x = logit_layer(hidden_output, W, b)

    # two functions for calculating the result and confidence/probability per class
    f_prob = theano.function([input_flatten], p_y_given_x)
    f_pred = theano.function([p_y_given_x], y_pred)

    result_prob = f_prob(filters_reshaped)
    result_pred = f_pred(result_prob)

    return result_pred, result_prob
