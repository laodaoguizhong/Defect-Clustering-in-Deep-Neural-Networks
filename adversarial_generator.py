import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, AutoAttack
from art.estimators.classification import PyTorchClassifier
import warnings
import time
warnings.filterwarnings('ignore')


class AdversarialSampleGenerator:
    def __init__(self, model, config, dataset_name):
        self.model = model.to(config.device).eval()
        self.config = config
        self.dataset_name = dataset_name
        self.device = config.device
        
        # 数据配置
        is_mnist = (dataset_name == 'mnist')
        self.input_shape = (1, 28, 28) if is_mnist else (3, 32, 32)
        self.batch_sizes = {
            'fgsm': 4096 if is_mnist else 512,
            'pgd': 2048 if is_mnist else 256,
            'deepfool': 1024 if is_mnist else 128,
            'autoattack': 32 if is_mnist else 16
        }
        
        # GPU优化
        torch.backends.cudnn.benchmark = True
        
        # ART分类器
        self.art_classifier = PyTorchClassifier(
            model=model, loss=nn.CrossEntropyLoss(),
            input_shape=self.input_shape, nb_classes=10,
            clip_values=(0.0, 1.0),
            device_type='gpu' if 'cuda' in str(self.device) else 'cpu'
        )
        
        print(f"\n{'='*60}")
        print(f"🚀 Adversarial Generator | {dataset_name.upper()} | {self.device}")
        print(f"{'='*60}\n")

    def _run_attack(self, name, x, y, attack_fn, batch_size):
        """通用攻击执行函数"""
        results = {'samples': [], 'indices': [], 'labels': [], 'adv_labels': [], 'success': 0, 'failed': 0}
        
        for start in tqdm(range(0, len(x), batch_size), desc=f"  {name}", ncols=80, leave=False):
            end = min(start + batch_size, len(x))
            x_batch, y_batch = x[start:end], y[start:end]
            
            try:
                x_adv = attack_fn(x_batch, y_batch)
                preds = self.art_classifier.predict(x_adv).argmax(axis=1)
                success_mask = (preds != y_batch)
                
                for i in np.where(success_mask)[0]:
                    results['samples'].append(x_adv[i])
                    results['indices'].append(start + i)
                    results['labels'].append(y_batch[i])
                    results['adv_labels'].append(preds[i])
                
                results['success'] += success_mask.sum()
                results['failed'] += (~success_mask).sum()
            except Exception:
                results['failed'] += len(x_batch)
        
        return results

    def generate_autoattack(self, x, y):
        """完整AutoAttack攻击 (APGD-CE, APGD-DLR, FAB, Square)"""
        eps = getattr(self.config, 'autoattack_eps', self.config.pgd_eps)
        
        attack = AutoAttack(
            estimator=self.art_classifier,
            norm='inf',
            eps=eps,
            eps_step=eps / 10,
            batch_size=self.batch_sizes['autoattack']
        )
        return attack.generate(x=x, y=y)

    def build_adversarial_library(self, data_loader, num_samples):
        """构建对抗样本库"""
        # 加载数据
        all_samples, all_labels = [], []
        for images, labels in data_loader:
            all_samples.append(images.numpy())
            all_labels.append(labels.numpy())
            if sum(s.shape[0] for s in all_samples) >= num_samples:
                break
        
        all_samples = np.concatenate(all_samples)[:num_samples]
        all_labels = np.concatenate(all_labels)[:num_samples]
        print(f"📥 Loaded {len(all_samples)} samples\n")
        
        # 初始化结果
        adv_library = {'samples': [], 'original_indices': [], 'attack_methods': [], 
                       'original_labels': [], 'adv_labels': []}
        stats = {}
        
        # 攻击配置
        attacks = [
            ('FGSM', [(f'fgsm_eps{eps}', lambda x, y, e=eps: FastGradientMethod(
                self.art_classifier, eps=e, batch_size=self.batch_sizes['fgsm']).generate(x, y)) 
                for eps in self.config.fgsm_eps]),
            ('PGD', [('pgd', lambda x, y: ProjectedGradientDescent(
                self.art_classifier, eps=self.config.pgd_eps, eps_step=self.config.pgd_alpha,
                max_iter=self.config.pgd_steps, batch_size=self.batch_sizes['pgd']).generate(x, y))]),
            ('DeepFool', [('deepfool', lambda x, y: DeepFool(
                self.art_classifier, max_iter=50, batch_size=self.batch_sizes['deepfool']).generate(x))]),
            ('AutoAttack', [('autoattack', self.generate_autoattack)])
        ]
        
        # 执行攻击
        for phase, attack_info in enumerate(attacks, 1):
            phase_name, attack_list = attack_info[0], attack_info[1]
            max_samples = attack_info[2] if len(attack_info) > 2 else len(all_samples)
            
            print(f"{'='*60}\n⚡ Phase {phase}/4: {phase_name}")

            print(f"{'='*60}")
            
            start_time = time.time()
            phase_success, phase_failed = 0, 0
            
            # 限制样本数量
            x_phase = all_samples[:max_samples]
            y_phase = all_labels[:max_samples]
            
            for method_name, attack_fn in attack_list:
                bs = self.batch_sizes.get(method_name.split('_')[0], self.batch_sizes['fgsm'])
                if 'autoattack' in method_name:
                    bs = self.batch_sizes['autoattack']
                
                res = self._run_attack(method_name, x_phase, y_phase, attack_fn, bs)
                
                for i, sample in enumerate(res['samples']):
                    adv_library['samples'].append(sample)
                    adv_library['original_indices'].append(res['indices'][i])
                    adv_library['attack_methods'].append(method_name)
                    adv_library['original_labels'].append(res['labels'][i])
                    adv_library['adv_labels'].append(res['adv_labels'][i])
                
                phase_success += res['success']
                phase_failed += res['failed']
            
            elapsed = time.time() - start_time
            stats[phase_name.lower()] = {'success': phase_success, 'failed': phase_failed, 'time': elapsed}
            print(f"✅ {phase_name}: {phase_success} success, {phase_failed} failed ({elapsed:.1f}s)\n")
        
        # 转换为numpy
        for key in adv_library:
            adv_library[key] = np.array(adv_library[key])
        
        # 统计输出
        print(f"{'='*60}")
        print(f"✅ Total: {len(adv_library['samples'])} adversarial samples")
        if len(adv_library['samples']) > 0:
            methods, counts = np.unique(adv_library['attack_methods'], return_counts=True)
            for m, c in zip(methods, counts):
                print(f"   {m:20s}: {c:5d} ({c/len(adv_library['samples'])*100:.1f}%)")
        print(f"{'='*60}\n")
        
        return adv_library
