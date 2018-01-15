from theano import gof, Op, tensor


def fmp(input, pool_ratio, constant, overlap=True):
    """fractional max pooling

    Parameters
    ----------
    input : N-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    pool_ratio : float
        Factor by which to downscale, usually in range (1,2)
    constant : float
        usually in range (0,1)
    overlap : bool
        When True, doing overlap fractional max pooling, otherwise disjoint
    """
    assert input.ndim == 4
    op = FMPool(pool_ratio, constant, overlap)
    output = op(input)
    return output


class FMPool(Op):
    """
    fractional max pooling c implement
    """
    __props__ = ('pool_ratio', 'constant', 'overlap')

    def __init__(self, pool_ratio, constant, overlap):
        self.pool_ratio = pool_ratio
        self.constant = constant
        self.overlap = overlap

    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError()
        x = tensor.as_tensor_variable(x)
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)
        return gof.Apply(self, [x], [out()])

    def c_headers(self):
        return ['<algorithm>']

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        fail = sub['fail']
        pool_ratio = self.pool_ratio
        constant = self.constant
        overlap = int(self.overlap)
        ccode = """
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int r = PyArray_DIMS(%(x)s)[2];
        int c = PyArray_DIMS(%(x)s)[3];
        const int z_r = (int)((float) r / %(pool_ratio)s);
        const int z_c = (int)((float) c / %(pool_ratio)s);
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        // memory allocation of z if necessary
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != z_r)
          ||(PyArray_DIMS(%(z)s)[3] != z_c)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_r;
          dims[3]=z_c;
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }
        int row_idx[z_r+1];
        for(int i=0; i<z_r+1; i++){
            row_idx[i] = (int)(%(pool_ratio)s * (i + %(constant)s));
        }
        int col_idx[z_c+1];
        for(int i=0; i<z_c+1; i++){
            col_idx[i] = (int)(%(pool_ratio)s * (i + %(constant)s));
        }
        // used for indexing a pool region inside the input
        int r_st, r_end, c_st, c_end;
        dtype_%(x)s collector; // temp var for the value in a region
        if (z_r && z_c)
        {
            for(int b=0; b<PyArray_DIMS(%(x)s)[0]; b++){
              for(int k=0; k<PyArray_DIMS(%(x)s)[1]; k++){
                for(int i=0; i< z_r; i++){
                  r_st = row_idx[i];
                  r_end = row_idx[i+1];
                  if (%(overlap)s) r_end += 1;
                  for(int j=0; j<z_c; j++){
                    c_st = col_idx[j];
                    c_end = col_idx[j+1];
                    if (%(overlap)s) c_end += 1;
                    dtype_%(z)s * z = (dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j));
                    // use the first element as the initial value of collector
                    collector = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s, b, k, r_st, c_st)))[0];
                    // go through the pooled region in the input
                    for(int m=r_st; m<r_end; m++)
                    {
                      if (m>=r) break;
                      for(int n=c_st; n<c_end; n++)
                      {
                        if (n>=c) break;
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s, b, k, m, n)))[0];
                        collector = (a > collector) ? a : collector;
                      }
                    }
                    z[0] = collector;
                  }
                }
              }
            }
        }
        """
        return ccode % locals()

    def c_code_cache_version(self):
        """
        The return value MUST BE CHANGED every time the C code is altered or else Theano will
        disregard the change in the code and simply load a previous version of the op from the cache.
        """
        return (0, 1, 1)


def fp(input, pool_ratio, constant, overlap=True, mode='max'):
    """fractional pooling

    Parameters
    ----------
    input : N-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    pool_ratio : float
        Factor by which to downscale, usually in range (1,2)
    constant : float
        usually in range (0,1)
    overlap : bool
        When True, doing overlap fractional max pooling, otherwise disjoint
    mode : {'max', 'sum', 'avg'}
        Operation executed on each window.
    """
    assert input.ndim == 4
    op = FPool(pool_ratio, constant, overlap, mode)
    output = op(input)
    return output


class FPool(Op):
    """
    fractional max pooling c implement
    """
    __props__ = ('pool_ratio', 'constant', 'overlap', 'mode')

    def __init__(self, pool_ratio, constant, overlap, mode):
        self.pool_ratio = pool_ratio
        self.constant = constant
        self.overlap = overlap
        self.mode = mode

    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError()
        x = tensor.as_tensor_variable(x)
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)
        return gof.Apply(self, [x], [out()])

    def c_headers(self):
        return ['<algorithm>']

    def c_code(self, node, name, inp, out, sub):
        if self.mode not in ('max', 'sum', 'avg'):
            raise theano.gof.utils.MethodNotDefined()
        x, = inp
        z, = out
        fail = sub['fail']
        pool_ratio = self.pool_ratio
        constant = self.constant
        overlap = int(self.overlap)
        ccode = """
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int r = PyArray_DIMS(%(x)s)[2];
        int c = PyArray_DIMS(%(x)s)[3];
        const int z_r = (int)((float) r / %(pool_ratio)s);
        const int z_c = (int)((float) c / %(pool_ratio)s);
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        // memory allocation of z if necessary
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != z_r)
          ||(PyArray_DIMS(%(z)s)[3] != z_c)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_r;
          dims[3]=z_c;
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }
        int row_idx[z_r+1];
        for(int i=0; i<z_r+1; i++){
            row_idx[i] = (int)(%(pool_ratio)s * (i + %(constant)s));
        }
        int col_idx[z_c+1];
        for(int i=0; i<z_c+1; i++){
            col_idx[i] = (int)(%(pool_ratio)s * (i + %(constant)s));
        }
        // used for indexing a pool region inside the input
        int r_st, r_end, c_st, c_end;
        dtype_%(x)s collector; // temp var for the value in a region
        if (z_r && z_c)
        {
            for(int b=0; b<PyArray_DIMS(%(x)s)[0]; b++){
              for(int k=0; k<PyArray_DIMS(%(x)s)[1]; k++){
                for(int i=0; i< z_r; i++){
                  r_st = row_idx[i];
                  r_end = row_idx[i+1];
                  if (%(overlap)s) r_end += 1;
                  for(int j=0; j<z_c; j++){
                    c_st = col_idx[j];
                    c_end = col_idx[j+1];
                    if (%(overlap)s) c_end += 1;
                    dtype_%(z)s * z = (dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j));
        """
        if self.mode == 'max':
            ccode += '''
                    // use the first element as the initial value of collector
                    collector = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s, b, k, r_st, c_st)))[0];
                    // go through the pooled region in the input
                    for(int m=r_st; m<r_end; m++)
                    {
                      if (m>=r) break;
                      for(int n=c_st; n<c_end; n++)
                      {
                        if (n>=c) break;
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s, b, k, m, n)))[0];
                        collector = (a > collector) ? a : collector;
                      }
                    }
                    z[0] = collector;
            '''
        elif self.mode in ('sum', 'avg'):
            ccode += '''
                    // initialize the sum at zero
                    collector = ((dtype_%(x)s)(0));
                    // go through the pooled region in the input
                    int r_flag = 0, c_flag = 0;
                    for(int m=r_st; m<r_end; m++)
                    {
                      if (m>=r){r_flag = 1; break;}
                      for(int n=c_st; n<c_end; n++)
                      {
                        if (n>=c){c_flag = 1; break;}
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s, b, k, m, n)))[0];
                        collector += a;
                      }
                    }
            '''
            if self.mode == 'sum':
                ccode += """
                    z[0] = collector;
                """
            else:
                ccode += """
                    int r_end_ = (r_flag == 0) ? r_end : r;
                    int c_end_ = (c_flag == 0) ? c_end : c;
                    z[0] = collector / ((r_end_-r_st)*(c_end_-c_st));
                """
        ccode += """
                  }
                }
              }
            }
        }
        """
        return ccode % locals()

    def c_code_cache_version(self):
        """
        The return value MUST BE CHANGED every time the C code is altered or else Theano will
        disregard the change in the code and simply load a previous version of the op from the cache.
        """
        return (0, 1, 1)


if __name__ == '__main__':
    import theano
    import theano.tensor as T
    import numpy as np

    # xt = T.tensor4()
    # poolx = fmp(xt, 1.414, 0.5)
    # pool = theano.function([xt], poolx, allow_input_downcast=True)
    # x = np.arange(400).reshape((2, 2, 10, 10))
    # print x, pool(x)

    xt = T.tensor4()
    poolx = fp(xt, 1.414, 0.5, mode='max')
    pool = theano.function([xt], poolx, allow_input_downcast=True)
    x = np.arange(400).reshape((2, 2, 10, 10))
    print x, pool(x)
