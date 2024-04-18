import torch.nn as nn
from layers import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list

class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')
        self.NUM_CLASSES = 4
        # self.embeddingsEnd = nn.Embedding(num_embeddings=self.NUM_CLASSES, embedding_dim=nr_filters) #end embeddeing
        self.embeddingsMiddleU = nn.Embedding(num_embeddings=self.NUM_CLASSES, embedding_dim=nr_filters) #middle embedding
        self.embeddingsMiddleUL = nn.Embedding(num_embeddings=self.NUM_CLASSES, embedding_dim=nr_filters) #begining embedding
        # self.embeddingsBeginingU = nn.Embedding(num_embeddings=self.NUM_CLASSES, embedding_dim=nr_filters) #begining embedding
        # self.embeddingsBeginingUL = nn.Embedding(num_embeddings=self.NUM_CLASSES, embedding_dim=nr_filters) #begining embedding

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None


    def forward(self, x, labels=None, sample=False):
        if labels == None:
            labels = self.predict(x, sample).to(x.device)

        # similar as done in the tf repo :
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        if not torch.is_tensor(labels):
            labels = torch.tensor(labels)
        # label_embeddingsEnd = self.embeddingsEnd(labels.to(x.device)).to(x.device)
        label_embeddingsMiddleU = self.embeddingsMiddleU(labels.to(x.device)).to(x.device)
        label_embeddingsMiddleUL = self.embeddingsMiddleUL(labels.to(x.device)).to(x.device)
        # label_embeddingsBeginingU = self.embeddingsBeginingU(labels.to(x.device)).to(x.device)
        # label_embeddingsBeginingUL = self.embeddingsBeginingUL(labels.to(x.device)).to(x.device)

        
        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]

        # u_list[-1] += label_embeddingsBeginingU.view(label_embeddingsBeginingU.shape[0], label_embeddingsBeginingU.shape[1], 1, 1).repeat(1, 1, u_list[-1].shape[2], u_list[-1].shape[3])
        # ul_list[-1] += label_embeddingsBeginingUL.view(label_embeddingsBeginingUL.shape[0], label_embeddingsBeginingUL.shape[1], 1, 1).repeat(1, 1, ul_list[-1].shape[2], ul_list[-1].shape[3])

        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        label_embeddingsMiddleU = label_embeddingsMiddleU.unsqueeze(-1).unsqueeze(-1)
        label_embeddingsMiddleU = label_embeddingsMiddleU.repeat(1, 1, u_list[-1].shape[2], u_list[-1].shape[3])
        
        label_embeddingsMiddleUL = label_embeddingsMiddleUL.unsqueeze(-1).unsqueeze(-1)
        label_embeddingsMiddleUL = label_embeddingsMiddleUL.repeat(1, 1, u_list[-1].shape[2], u_list[-1].shape[3])
        u_list[-1] += label_embeddingsMiddleU
        ul_list[-1] += label_embeddingsMiddleUL

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        #reshape the label embeddings
        # label_embeddingsEnd = label_embeddingsEnd.view(label_embeddingsEnd.shape[0], label_embeddingsEnd.shape[1], 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
        # ul = label_embeddingsEnd + ul
        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
    
    def predict(self, x, sample=False):
        x_new = x.repeat(self.NUM_CLASSES, 1, 1, 1)
        labels = torch.arange(self.NUM_CLASSES).repeat_interleave(x.shape[0])
        losses = self.forward(x_new, labels=labels, sample=sample)
        losses = discretized_mix_logistic_loss_batch(x_new, losses)
        losses = losses.view(self.NUM_CLASSES, -1).permute(1, 0)
        return torch.argmin(losses, dim=1)
    
    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if 'models' not in os.listdir():
            os.mkdir('models')
        torch.save(self.state_dict(), 'models/conditional_pixelcnn.pth')
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)
    
