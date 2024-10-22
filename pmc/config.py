import torch, importlib, pathlib
from torch.utils.data import DataLoader
from fvcore.common.config import CfgNode as CN

class Configurator:
    def __init__(self, args):
        """
        Create configs and make fixes
        """
        self.cfg = CN(CN.load_yaml_with_base(args.config))
        self.cfg = self._default_fix(self.cfg)
        self.cfg.freeze()

    def init_all(self):
        # model (pmc)
        model = self._init_model()
        # dataloader
        dataloader = self._init_dataloader()
        # callbacks
        callbacks = self._init_callbacks()
        # experiment
        exp_ = self.str_to_class('pmc.experiments', self.cfg.procedure.name)
        exp = exp_(self.cfg)

        return exp, model, dataloader, callbacks


    ##########################################
    ###          Model Selection           ###
    ##########################################

    def _default_fix(self, cfg):
        # set save_dir & names
        # mid_dir = "_".join(cfg.exp_name.split("_", 3)[:3])
        cfg.exp_dir = f"./results/{cfg.exp_name}"
        cfg.logger.name = cfg.exp_name
        cfg.logger.dir = cfg.exp_dir

        # set accelerator
        if self.cfg.accelerator == 'gpu':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)

        return cfg

    def _init_model(self):
        # initialize submodules
        forward_model = self.init_params_without_name("pmc.forward_models", self.cfg.model.forward_model)
        coeff = self.init_params_without_name("pmc.coeffs", self.cfg.model.coeff)
        if 'transform' in self.cfg.model.keys():
            transform = self.init_params_without_name("pmc.transforms", self.cfg.model.transform)
        else:
            transform = None
        # initialize score_fn 
        score_fn = self.init_params_without_name("pmc.score_fns", self.cfg.model.score_fn)
        # load checkpoint if given
        if 'init_score_fn_dir' in self.cfg.checkpoint.keys():
            ckpt = self._init_ckpt(self.cfg.checkpoint.init_score_fn_dir)
            score_fn = self._load_model_from_ckpt(ckpt, score_fn)

        # initialize model
        model_ob = self.str_to_class('pmc.algorithms', self.cfg.model.name)
        init_dict = self.cfg_to_dict_without_sth(self.cfg.model, ['name', 'forward_model', 'score_fn', 'coeff', 'transform'])
        return model_ob(forward_model, score_fn, coeff, transform=transform, **init_dict)

    def _init_dataloader(self):
        dataset = self.init_params_without_name("pmc.test_datasets", self.cfg.dataset)
        return DataLoader(dataset=dataset, **dict(self.cfg.dataloader))

    def _init_callbacks(self):
        callbacks = []
        if 'callbacks' in self.cfg.keys():
            for cb_cfg in self.cfg.callbacks.values():
                callbacks.append(self.init_params_without_name("pmc.callbacks", cb_cfg))
        else:
            callbacks = None
        return callbacks


    ##########################################
    ###          Load Checkpoints          ###
    ##########################################

    def _init_ckpt(self, exp_dir):
        # return if exp_dir points to a file
        exp_dir = pathlib.Path(exp_dir)
        if exp_dir.is_file():
            return exp_dir

        # loop over folder
        ckpt_list = []
        for fname in list(exp_dir.iterdir()):
            if fname.name[-5:] == '.ckpt' or \
                fname.name[-4:] == '.pth' or \
                fname.name[-3:] == '.pt':
                ckpt_list.append(fname)
        
        if len(ckpt_list) == 0:
            raise FileNotFoundError('There is no checkpoint in the directory!')
        elif len(ckpt_list) > 1:
            raise RuntimeError('There are multiple checkpoints in the directory!')

        return ckpt_list[0]

    def _load_model_from_ckpt(self, ckpt, model):
        # state dict of the model for initialization
        if ckpt.name[-5:] == '.ckpt':
            if self.cfg.checkpoint.load_ema:
                print('loaded ema weights')
                model_dict = torch.load(ckpt)['state_dict_ema']
            else:
                print('loaded original weights')
                model_dict = torch.load(ckpt)['state_dict']
            model_dict = self.remove_prefix(model_dict)
        elif ckpt.name[-4:] == '.pth':
            model_dict = torch.load(ckpt)
        elif ckpt.name[-3:] == '.pt':
            model_dict = torch.load(ckpt)
        else:
            raise RuntimeError('model checkpoint should be .ckpt or .pth or .pt')

        # initialize
        model.load_state_dict(model_dict, strict=True)

        # set trainability of initialized modules
        for param in model.parameters():
            param.requires_grad = self.cfg.checkpoint.score_fn_trainability

        return model

    ##########################################
    ###          Static Methods            ###
    ##########################################

    @staticmethod
    def remove_prefix(dict, num_prefix=1):
        """ Remove the given prefix """
        out_dict = {}
        for key,val in dict.items():
            splits = key.split('.')
            out_key = '.'.join(splits[num_prefix:])
            out_dict[out_key] = val
        return out_dict

    @staticmethod
    def str_to_class(module_name, class_name):
        """ Return a class instance from a string reference """
        try:
            module_ = importlib.import_module(module_name)
            try:
                class_ = getattr(module_, class_name)
            except AttributeError:
                raise AttributeError(f'Class [{class_name}] does not exist')
        except ImportError:
            raise ImportError(f'Module [{module_name}] does not exist')
        return class_

    @staticmethod
    def init_params_without_name(module_name, cfg):
        class_ = Configurator.str_to_class(module_name, cfg.name)
        init_dict = dict(cfg)
        del init_dict["name"]
        return class_(**init_dict)

    @staticmethod
    def cfg_to_dict_without_sth(cfg, sth_list):
        init_dict = dict(cfg)
        for sth in sth_list:
            try:
                del init_dict[sth]
            except:
                print(f'name [{sth}] does not exist in cfg. Pass...')
        return init_dict

    @staticmethod
    def cfg_to_dict_without_name(cfg):
        init_dict = dict(cfg)
        del init_dict["name"]
        return init_dict