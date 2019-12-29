from __future__ import division
from conv_layer import BaseConvLayer
from collections import OrderedDict
import numpy as np

dtype = np.float32


class PoolLayer(BaseConvLayer):
    def __init__(self, act_type, kernel_size, stride, pad):
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.outputs = []

        self.params = OrderedDict()
        super(PoolLayer, self).__init__()

    def init(self, height, width, in_channels):
        """No need to implement this func"""
        pass

    def forward(self, inputs):
        """The forward pass

        Arguments:
            inputs (OrderedDict): A dictionary containing
                height: the height of the current input
                width: the width of the current input
                channels: number of channels in the current
                    inputs
                data: a flattened data array of the form
                    batch_size * height * width * channel, unrolled in
                    the same way
        Returns:
            outputs (OrderedDict): A dictionary containing
                height: The height of the output
                width: The width of the output
                out_channels: The output number of feature maps
                    Same as the input channels for this layer
                data: a flattened output data array of the form
                    height * width * channel, unrolled in
                    the same way

        You may want to take a look at the im2col_conv and col2im_conv
        functions present in the base class ``BaseConvLayer``

        You may also find it useful to cache the height, width and
        channel of the input image for the backward pass.
        The output heights, widths and channels can be computed
        on the fly using the ``get_output_dim`` function.

        """
        h_in = inputs["height"]
        w_in = inputs["width"]
        c = inputs["channels"]
        data = inputs["data"]
        batch_size = data.shape[0]
        k = self.kernel_size
        h_out, w_out, c = self.get_output_dim(
            h_in, w_in, self.pad, self.stride,
            self.kernel_size, c
        )

        # cache for backward pass
        self.h_in, self.w_in, self.c = h_in, w_in, c
        self.data = data

        outputs = OrderedDict()
        outputs["height"] = h_out
        outputs["width"] = w_out
        outputs["channels"] = c

        outputs["data"] = np.zeros((batch_size, h_out * w_out * c), dtype=dtype)
        ##################
        ## Using im2col ##
        ##################
        stack = []
        for ix in range(batch_size):
            col = self.im2col_conv(data[ix], h_in, w_in, c, h_out, w_out).reshape((k*k, c, h_out*w_out))
            self.outputs.append(col)
            out = np.max(col, axis = 0).transpose(1, 0).flatten()
            stack.append(out)
        stack = np.stack(stack)
        outputs["data"] = stack

        # outputs["data"] = np.zeros(
        # (batch_size, h_out * w_out * c), dtype=dtype).reshape((batch_size, h_out, w_out, c))
        # outputs["data"] = data.reshape(batch_size, c, h_out, k, w_out, k).max(axis=(3,5))
        #outputs["data"] = outputs["data"].flatten().reshape((batch_size, h_out*w_out*c))
        
        return outputs

    def backward(self, output_grads):
        """The backward pass

        Arguments:
            output_grads (OrderedDict): Containing
                grad: gradient wrt output
        Returns:
            input_grads (OrderedDict): Containing
                grad: gradient wrt input

        Note that we compute the output heights, widths, and
        channels on the fly in the backward pass as well.

        You may want to take a look at the im2col_conv and col2im_conv
        functions present in the base class ``BaseConvLayer``

        """
        input_grads = OrderedDict()
        input_grads["grad"] = np.zeros_like(self.data, dtype=dtype)
        h_in, w_in, c = self.h_in, self.w_in, self.c
        batch_size = self.data.shape[0]
        output_diff = output_grads["grad"]

        k = self.kernel_size
        stride = self.stride
        p = self.pad

        h_out, w_out, c = self.get_output_dim(
            h_in, w_in, self.pad, self.stride,
            self.kernel_size, c
        )

        data = self.data
        outputs = self.outputs
        ##################
        ## Using col2im ##
        ##################
        for ix in range(batch_size):
            col = outputs[ix].reshape(c, k*k, h_out*w_out)
            out = np.max(col, axis = 0, keepdims = True)
            max_i = np.equal(out, col).transpose(1, 0, 2)

            # col = self.im2col_conv(data[ix], h_in, w_in, c, h_out, w_out).flatten().reshape((k,k, c, h_out*w_out))
            # col = col.flatten().reshape((k*k, c, h_out*w_out))
            # max_i = np.argmax(col, axis = 0)
            # mask = np.zeros((k*k, c, h_out*w_out))
            # mask[max_i] = 1

            grads = output_diff[ix].reshape(h_out*w_out, c).transpose(1, 0)
            new_col = grads[np.newaxis, :, :] * max_i
            # grads = np.repeat(grads[np.newaxis, :, :], k*k, axis=0)
            # col = np.multiply(grads, mask).flatten().reshape(k*k*c, h_out*w_out)
            im = self.col2im_conv(new_col, h_in, w_in, c, h_out, w_out)
            input_grads["grad"][ix] = im.flatten()


        ##################################
        ## Using repeat and np.multiply ##
        ##################################
        # col = np.zeros_like(k*k*c, h_out*w_out)
        # grads_flat = output_diff.reshape((batch_size, h_in, w_in, c)transpose(2, 3, 0, 1).ravel()

        # col[max_i, range(max_i.size)] = grads_flat
        # grads = self.col2im_conv(col, h_in, w_in, c, h_out, w_out)
        # input_grads["grad"] = grads.reshape(self.data.shape)
        # print(np.sum(input_grads["grad"]))
        # # create array filled with grad values and mask
        # maxvals = np.repeat(self.outputs["data"], k**2, axis = 1)
        # mask = np.equal(data,maxvals)
        # #print(mask)
        # grads = np.repeat(output_diff, k**2, axis = 1)
        # # grads = np.repeat(np.repeat(grads, k, axis=2), k, axis=3)
        # for ix in range(batch_size):
        #     input_grads["grad"][ix] = np.multiply(mask[ix],grads[ix])

        #################
        ## Using loops ##
        #################
        # input_grads["grad"] = input_grads["grad"].reshape((batch_size, h_in, w_in, c))
        # for ix in range(batch_size):
        #     im = data[ix].flatten().reshape((h_in, w_in, c))
        #     tmp_diff = output_diff[ix].flatten().reshape((h_out*w_out, c))
        #     for depth in range(c):
        #         for h in range(h_out):
        #             for w in range(w_out):
        #                 left_r = h * stride
        #                 right_r = h * stride + k
        #                 left_c = w * stride
        #                 right_c = w * stride + k
        #                 pool = im[left_r:right_r, left_c:right_c, depth]
        #                 mask = (pool == np.max(pool))
        #                 input_grads["grad"][ix][left_r:right_r, left_c:right_c, depth] = \
        #                     mask*tmp_diff[h*w_out+w,depth]
        # input_grads["grad"] = input_grads["grad"].flatten().reshape((batch_size, h_in*w_in*c))

        return input_grads
