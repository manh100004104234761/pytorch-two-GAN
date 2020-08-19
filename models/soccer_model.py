from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks


class SoccerModel(BaseModel):
    def name(self):
        return 'SoccerModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.seg_netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                          opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.detec_netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                            opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.load_network(self.seg_netG, 'G', 'seg_latest')
        self.load_network(self.detec_netG, 'G', 'detec_latest')
        print('Warning: continue_train is not supported')
        print('---------- Networks initialized -------------')
        networks.print_network(self.seg_netG)
        networks.print_network(self.detec_netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.seg_netG(self.real_A)
        fake_B = (self.fake_B + 1.0)/2.0
        input_A = (self.real_A + 1.0)/2.0
        self.fake_C = (fake_B * input_A) * 2.0 - 1
        self.fake_D = self.detec_netG(self.fake_C)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        fake_D = util.tensor2im(self.fake_D.data)
        fake_C = util.tensor2im(self.fake_C.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('fake_C', fake_C),
                            ('fake_D', fake_D)])
