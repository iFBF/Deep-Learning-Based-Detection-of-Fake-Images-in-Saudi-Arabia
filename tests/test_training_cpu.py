import importlib.util,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional CPU training checks require PyTorch')
class TrainingCPU(unittest.TestCase):
    def test_checkpoint_uses_lowest_loss_even_below_early_stop_delta(self):
        import torch,pandas as pd,numpy as np
        import phase1_training as training
        torch.set_num_threads(2)
        frame=pd.DataFrame({'relative_path':[str(i) for i in range(8)],'label':[0,0,1,1,0,0,1,1],'split':['train']*4+['val']*4})
        def small_loader(rows,root,mode,config,training=False,corruption=None):
            x=torch.rand((len(rows),3,32,32),generator=torch.Generator().manual_seed(9));y=torch.tensor(rows.label.to_numpy())
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=2)
        scores=np.array([.3,.4,.6,.7])
        with tempfile.TemporaryDirectory() as directory:
            config={**training.DEFAULTS,'cnn_epochs':2,'batch_size':2,'min_delta':.0001}
            with patch.object(training,'loader',small_loader),patch.object(training,'validation_predictions',side_effect=[(scores,.7),(scores,.69999)]):
                result=training.train('scratch',frame,'.',directory,config)
            self.assertEqual(result['epoch'],2)
            state=torch.load(Path(directory)/'scratch/best.pt',map_location='cpu',weights_only=True)
            self.assertEqual(state['epoch'],2)
    @unittest.skipUnless(importlib.util.find_spec('transformers'),'Optional transfer checks require Transformers')
    def test_transfer_checkpoint_and_processor_local_roundtrips(self):
        import torch,pandas as pd
        from transformers import SwinConfig,AutoModelForImageClassification,AutoImageProcessor,ViTImageProcessor
        import phase1_training as training
        torch.set_num_threads(2)
        frame=pd.DataFrame({'relative_path':[str(i) for i in range(8)],'label':[0,0,1,1,0,0,1,1],'split':['train']*4+['val']*4})
        initial={}
        def small_model(mode):
            config=SwinConfig(image_size=32,patch_size=4,embed_dim=8,depths=[1,1],num_heads=[1,2],window_size=2,num_labels=2)
            model=AutoModelForImageClassification.from_config(config)
            for p in model.parameters():p.requires_grad=(mode=='full')
            for p in model.classifier.parameters():p.requires_grad=True
            initial[mode]={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('classifier.')}
            return model
        def small_loader(rows,root,mode,config,training=False,corruption=None):
            x=torch.rand((len(rows),3,32,32),generator=torch.Generator().manual_seed(9));y=torch.tensor(rows.label.to_numpy())
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=2)
        processor=ViTImageProcessor(size={'height':224,'width':224},image_mean=[.485,.456,.406],image_std=[.229,.224,.225],resample=3)
        with tempfile.TemporaryDirectory() as directory:
            for mode in ['frozen','full']:
                config={**training.DEFAULTS,'transfer_epochs':1,'batch_size':2}
                with patch.object(training,'build_model',small_model),patch.object(training,'loader',small_loader),patch.object(AutoImageProcessor,'from_pretrained',return_value=processor):
                    training.train(mode,frame,'.',directory,config)
                restored=training.load_checkpoint(mode,Path(directory)/mode,'cpu')
                with torch.inference_mode():output=restored(pixel_values=torch.rand(1,3,32,32)).logits
                self.assertEqual(tuple(output.shape),(1,2))
                loaded_processor=AutoImageProcessor.from_pretrained(str(Path(directory)/mode/'processor'),local_files_only=True)
                self.assertEqual(loaded_processor.size['height'],224)
                same=[torch.equal(v,restored.state_dict()[k]) for k,v in initial[mode].items()]
                self.assertEqual(all(same),mode=='frozen')
    def test_weighting_does_not_cancel_at_batch_one(self):
        import torch
        x=torch.tensor([[.2,-.1]],requires_grad=True)
        loss=torch.nn.functional.cross_entropy(x,torch.tensor([0]),reduction='none').sum()
        loss.backward();gradient=x.grad.clone();x.grad.zero_()
        weighted=(torch.nn.functional.cross_entropy(x,torch.tensor([0]),reduction='none')*2).sum()
        weighted.backward();self.assertTrue(torch.allclose(x.grad,2*gradient))
    def test_real_scratch_training_exports_selected_metrics_and_weights(self):
        import torch,pandas as pd,numpy as np,json
        import phase1_training as training
        from phase1_corrections import prediction_metrics
        torch.set_num_threads(2)
        frame=pd.DataFrame({'relative_path':[str(i) for i in range(12)],
            'label':[0,0,1,1,1,1,1,1,0,0,1,1],'split':['train']*8+['val']*4})
        def synthetic_loader(rows,root,mode,config,training=False,corruption=None):
            x=torch.rand((len(rows),3,32,32),generator=torch.Generator().manual_seed(44))
            y=torch.tensor(rows.label.to_numpy(),dtype=torch.long)
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=config['batch_size'])
        with tempfile.TemporaryDirectory() as directory:
            config={**training.DEFAULTS,'cnn_epochs':1,'batch_size':1,'accumulation_steps':3}
            with patch.object(training,'loader',synthetic_loader):
                result=training.train('scratch',frame,'.',directory,config)
            path=Path(directory)/'scratch';model=training.load_checkpoint('scratch',path,'cpu')
            self.assertEqual(sum(p.numel() for p in model.parameters()),389410)
            self.assertEqual(result['epoch'],1)
            predictions=pd.read_csv(path/'selected_validation_predictions.csv')
            self.assertAlmostEqual(prediction_metrics(predictions)['macro_f1'],result['val_macro_f1'])
            weights=json.loads((path/'training_class_weights.json').read_text())['weights']
            self.assertTrue(np.allclose(weights,[2.,2/3]))

if __name__=='__main__':unittest.main()
