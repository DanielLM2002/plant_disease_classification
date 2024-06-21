import torch.nn as nn

# Model parameters:
LAYERS = 5
KERNELS = [3, 3, 3, 3, 3]
CHANNELS = [32, 64, 128, 256, 512]
STRIDES = [2, 2, 2, 2, 2]
LINEAR_DIM = 512 * 7 * 7


class Encoder(nn.Module):
    
    def __init__(self, output_dim=2, use_batchnorm=False, use_dropout=False):
        super(Encoder, self).__init__()
        
        # bottleneck dimentionality
        self.output_dim = output_dim

        # variables deciding if using dropout and batchnorm in model
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
    
        # convolutional layer hyper parameters
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS
        self.strides = STRIDES
        self.conv = self.get_convs()
        
        # layers for latent space projection
        self.fc_dim = LINEAR_DIM
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.fc_dim, self.output_dim)
    
    
    def get_convs(self):
        """
        generating convolutional layers based on model's 
        hyper parameters
        """
        conv_layers = nn.Sequential()
        for i in range(self.layers):
            # The input channel of the first layer is 1
            if i == 0: conv_layers.append(nn.Conv2d(1, 
                                              self.channels[i], 
                                              kernel_size=self.kernels[i],
                                              stride=self.strides[i],
                                              padding=1))
            
            else: conv_layers.append(nn.Conv2d(self.channels[i-1], 
                                         self.channels[i],
                                         kernel_size=self.kernels[i],
                                         stride=self.strides[i],
                                         padding=1))
            
            if self.use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(self.channels[i]))
            
            # Here we use GELU as activation function
            conv_layers.append(nn.GELU()) 
            
            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))

        return conv_layers
    
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.linear(x)
    

class Decoder(nn.Module):
    
    def __init__(self, input_dim=2, use_batchnorm=False, use_dropout=False):
        
        super(Decoder, self).__init__()
        
        # variables deciding if using dropout and batchnorm in model
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.fc_dim = LINEAR_DIM
        self.input_dim = input_dim
        
        # Conv layer hypyer parameters
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS[::-1] # flip the channel dimensions
        self.strides = STRIDES
        
        # In decoder, we first do fc project, then conv layers
        self.linear = nn.Linear(self.input_dim, self.fc_dim)
        self.conv =  self.get_convs()

        self.output = nn.Conv2d(self.channels[-1], 1, kernel_size=1, stride=1)

    def get_convs(self):
        conv_layers = nn.Sequential()
        for i in range(self.layers):
            
            if i == 0: conv_layers.append(
                            nn.ConvTranspose2d(self.channels[i],
                                               self.channels[i],
                                               kernel_size=self.kernels[i],
                                               stride=self.strides[i],
                                               padding=1,
                                               output_padding=1)
                            )
            
            else: conv_layers.append(
                            nn.ConvTranspose2d(self.channels[i-1], 
                                               self.channels[i],
                                               kernel_size=self.kernels[i],
                                               stride=self.strides[i],
                                               padding=1,
                                               output_padding=1
                                              )
                            )
            
            if self.use_batchnorm and i != self.layers - 1:
                conv_layers.append(nn.BatchNorm2d(self.channels[i]))

            conv_layers.append(nn.GELU())

            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))

        return conv_layers
    
    
    def forward(self, x):
        x = self.linear(x)
        # reshape 3D tensor to 4D tensor
        x = x.reshape(x.shape[0], 512, 7, 7)
        x = self.conv(x)
        return (self.output(x))


class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(output_dim=4096, 
                               use_batchnorm=True, use_dropout=False)
        self.decoder = Decoder(input_dim=4096,
                               use_batchnorm=True, use_dropout=False)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))