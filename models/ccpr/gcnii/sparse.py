import torch as th
from ...base import is_all, ALL
from ...sparse import _gspmm, _gsddmm

__all__ = ['gspmm', 'gsddmm', 'edge_softmax']
# CONFIGURE S here, will be used in GSpMM in line 62
S = 256

def _reduce_grad(grad, shape):
    """Reduce gradient on the broadcast dimension
    If there is broadcast in forward pass, gradients need to be reduced on
    broadcast dimension. This function checks the input tensor shape and
    gradient shape and perform the reduction.

    Parameters
    ----------
    grad: Tensor
        Gradient tensor
    shape: tuple
        Shape of input tensor

    Returns
    -------
    Tensor
    """
    grad_shape = grad.shape[1:]
    in_shape = shape[1:]
    if in_shape == grad_shape:
        # no need to reduce
        return grad
    num_to_squeeze = len(grad_shape) - len(in_shape)
    # pad inshape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = th.nonzero(th.tensor(grad_shape) - th.tensor(in_shape))
    reduce_idx += 1  # skip batch dim
    if len(reduce_idx) > 0:
        grad = grad.sum(dim=tuple(reduce_idx), keepdim=True)
    return grad.view(-1, *shape[1:])


def _need_reduce_last_dim(ufeat, efeat):
    """Indicates whether to reduce the last dimension on edges
    in the backward pass of spmm,
    if so, use dot instead of mul."""
    ushp = ufeat.shape
    eshp = efeat.shape
    return ushp[1:-1] == eshp[1:-1] and eshp[-1] == 1 and ushp[-1] > 1


def _muldiv(op, x):
    return 1. / x if op == 'div' else x


def _addsub(op, x):
    return -x if op == 'sub' else x


def _expand(x, shape):
    return x.expand(-1, *shape)


class GSpMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, gidx, op, reduce_op, X, Y):
        out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y, S)
        ctx.backward_cache = gidx, op, reduce_op
        ctx.save_for_backward(X, Y, argX, argY)
        return out

    @staticmethod
    def backward(ctx, dZ):
        gidx, op, reduce_op = ctx.backward_cache
        X, Y, argX, argY = ctx.saved_tensors
        if op != 'copy_rhs' and ctx.needs_input_grad[3]:
            g_rev = gidx.reverse()
            if reduce_op == 'sum':
                if op in ['mul', 'div']:
                    dX = gspmm(g_rev, 'mul', 'sum', dZ, _muldiv(op, Y))
                elif op in ['add', 'sub']:
                    dX = gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)
                elif op == 'copy_lhs':
                    dX = gspmm(g_rev, 'copy_lhs', 'sum', dZ, None)
            else:  # max/min
                dX = th.zeros((X.shape[0],) + dZ.shape[1:],
                              dtype=X.dtype, device=X.device)
                if op in ['mul', 'div']:
                    grad = _muldiv(op, _expand(Y, dZ.shape[1:]).gather(
                        0, argY.long())) * dZ
                    dX.scatter_add_(0, argX.long(), grad)
                elif op in ['add', 'sub', 'copy_lhs']:
                    dX.scatter_add_(0, argX.long(), dZ)
            dX = _reduce_grad(dX, X.shape)
        else:  # X has not gradient
            dX = None
        if op != 'copy_lhs' and ctx.needs_input_grad[4]:
            if reduce_op == 'sum':
                if op == 'mul' and _need_reduce_last_dim(X, Y):
                    dY = gsddmm(gidx, 'dot', X, dZ)
                elif op in ['mul', 'div']:
                    dY = gsddmm(gidx, 'mul', X, dZ)
                    if op == 'div':
                        dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY = gsddmm(gidx, 'copy_rhs', X, _addsub(op, dZ))
            else:  # max/min
                dY = th.zeros((Y.shape[0],) + dZ.shape[1:],
                              dtype=Y.dtype, device=Y.device)
                if op in ['mul',  'div']:
                    grad = _expand(X, dZ.shape[1:]).gather(
                        0, argX.long()) * dZ
                    dY.scatter_add_(0, argY.long(), grad)
                    if op == 'div':
                        dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY.scatter_add_(0, argY.long(), _addsub(op, dZ))
            dY = _reduce_grad(dY, Y.shape)
        else:  # Y has no gradient
            dY = None
        return None, None, None, dX, dY


class GSDDMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, gidx, op, X, Y, lhs_target, rhs_target):
        out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)
        ctx.backward_cache = gidx, op, lhs_target, rhs_target
        ctx.save_for_backward(X, Y)
        return out

    @staticmethod
    def backward(ctx, dZ):
        gidx, op, lhs_target, rhs_target = ctx.backward_cache
        X, Y = ctx.saved_tensors
        if op != 'copy_rhs' and ctx.needs_input_grad[2]:
            if lhs_target in ['u', 'v']:
                _gidx = gidx if lhs_target == 'v' else gidx.reverse()
                if op in ['add', 'sub', 'copy_lhs']:
                    dX = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)
                else:  # mul, div, dot
                    if rhs_target == lhs_target:
                        dX = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ) * _muldiv(op, Y)
                    elif rhs_target == 'e':
                        dX = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ * _muldiv(op, Y))
                    else:  # rhs_target = !lhs_target
                        dX = gspmm(_gidx, 'mul', 'sum', _muldiv(op, Y), dZ)
            else:  # lhs_target == 'e'
                if op in ['add', 'sub', 'copy_lhs']:
                    dX = dZ
                else:  # mul, div, dot
                    dX = gsddmm(gidx, 'mul', dZ, _muldiv(op, Y), 'e', rhs_target)
            dX = _reduce_grad(dX, X.shape)
        else:
            dX = None
        if op != 'copy_lhs' and ctx.needs_input_grad[3]:
            if rhs_target in ['u', 'v']:
                _gidx = gidx if rhs_target == 'v' else gidx.reverse()
                if op in ['add', 'sub', 'copy_rhs']:
                    dY = gspmm(_gidx, 'copy_rhs', 'sum', None, _addsub(op, dZ))
                else:  # mul, div, dot
                    if lhs_target == rhs_target:
                        dY = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ) * X
                    elif lhs_target == 'e':
                        dY = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ * X)
                    else:  # rhs_target = !lhs_target
                        dY = gspmm(_gidx, 'mul', 'sum', X, dZ)
                    if op == 'div':
                        dY = -dY / (Y ** 2)
            else:
                if op in ['add', 'sub', 'copy_rhs']:
                    dY = _addsub(op, dZ)
                else:  # mul, div, dot
                    dY = gsddmm(gidx, 'mul', dZ, X, 'e', lhs_target)
                    if op == 'div':
                        dY = -dY / (Y ** 2)
            dY = _reduce_grad(dY, Y.shape)
        else:
            dY = None
        return None, None, dX, dY, None, None


class EdgeSoftmax(th.autograd.Function):
    @staticmethod
    def forward(ctx, gidx, score, eids, norm_by):
        """Forward function.

        Pseudo-code:

        .. code:: python

            score = dgl.EData(g, score)
            score_max = score.dst_max()  # of type dgl.NData
            score = score - score_max  # edge_sub_dst, ret dgl.EData
            score_sum = score.dst_sum()  # of type dgl.NData
            out = score / score_sum    # edge_div_dst, ret dgl.EData
            return out.data
        """
        # remember to save the graph to backward cache before making it
        # a local variable
        if not is_all(eids):
            gidx = gidx.edge_subgraph([eids], True).graph
        if norm_by == 'src':
            gidx = gidx.reverse()
        score_max = _gspmm(gidx, 'copy_rhs', 'max', None, score)[0]
        score = th.exp(_gsddmm(gidx, 'sub', score, score_max, 'e', 'v'))
        score_sum = _gspmm(gidx, 'copy_rhs', 'sum', None, score)[0]
        out = _gsddmm(gidx, 'div', score, score_sum, 'e', 'v')
        ctx.backward_cache = gidx
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Backward function.

        Pseudo-code:

        .. code:: python

            g, out = ctx.backward_cache
            grad_out = dgl.EData(g, grad_out)
            out = dgl.EData(g, out)
            sds = out * grad_out  # type dgl.EData
            sds_sum = sds.dst_sum()  # type dgl.NData
            grad_score = sds - out * sds_sum  # multiple expressions
            return grad_score.data
        """
        gidx = ctx.backward_cache
        out, = ctx.saved_tensors
        sds = out * grad_out
        accum = gspmm(gidx, 'copy_rhs', 'sum', None, sds)
        grad_score = sds - gsddmm(gidx, 'mul', out, accum, 'e', 'v')
        return None, grad_score, None, None


def gspmm(gidx, op, reduce_op, lhs_data, rhs_data):
    return GSpMM.apply(gidx, op, reduce_op, lhs_data, rhs_data)


def gsddmm(gidx, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v'):
    return GSDDMM.apply(gidx, op, lhs_data, rhs_data, lhs_target, rhs_target)


def edge_softmax(gidx, logits, eids=ALL, norm_by='dst'):
    return EdgeSoftmax.apply(gidx, logits, eids, norm_by)
