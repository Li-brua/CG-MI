from losses.poincare import poincare_loss, max_margin_loss, CE_loss


import numpy as np
import torch
from attacks.turbo import Turbo1, TurboM
import nevergrad as ng
from tqdm import tqdm



class Optimization():
    def __init__(self, target_model, synthesis, mapping, z_dim, transformations,
                 config, device):
        self.generator = synthesis
        self.mapping = mapping
        self.target = target_model
        self.z_dim = z_dim
        self.config = config
        self.transformations = transformations
        self.clip = config.attack['clip']
        self.use_pbar = False
        self.optimize_method = self.config.attack['optimizer']
        self.device = device

    def optimize(self, target, init_w=None, result_dict=None, rank=None, lock=None):
        if self.optimize_method == "BO":
            # Initialize attack
            z_lb = -2 * np.ones(self.z_dim)  # lower bound, you may change -10 to -inf
            z_ub = 2 * np.ones(self.z_dim)  # upper bound, you may change 10 to inf

            f = lambda x: self.evaluate_loss(x, target)
            self.optimizer = Turbo1(
                f=f,  # Handle to objective function
                lb=z_lb,  # Numpy array specifying lower bounds
                ub=z_ub,  # Numpy array specifying upper bounds
                n_init=self.config.attack["BO"]["n_init"],  # Number of initial bounds from an Latin hypercube design
                max_evals=self.config.attack["BO"]["max_evals"],  # Maximum number of evaluations
                # n_trust_regions=self.config.attack["BO"]["n_trust_regions"],  # Number of regions to divide the domain
                batch_size=self.config.attack["BO"]["batch_size"],  # How large batch size TuRBO uses
                verbose=True,  # Print information from each batch
                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                n_training_steps=100,  # Number of steps of ADAM to learn the hypers
                min_cuda=1024,  # Run on the CPU for small datasets
                device="cpu",  # next(generator.parameters()).device,  # "cpu" or "cuda"
                dtype="float32",  # float64 or float32
            )

            self.optimizer.optimize()
            X = self.optimizer.X  # Evaluated points w
            fX = self.optimizer.fX  # Observed losses

            w_batch = []
            for batch_num in range(0, self.config.attack["BO"]["number_samples"]):
                ind_best_batch = np.argmin(fX)
                w_batch.append(torch.from_numpy(X[ind_best_batch, :]).unsqueeze(0))
                # remove all value equal to fX[ind_best_batch] in fX
                fX = np.delete(fX, ind_best_batch)
            w_batch = torch.cat(w_batch, dim=0)

            with lock:
                if target.item() in result_dict:
                    result_dict[target.item()] = torch.cat((result_dict[target.item()], w_batch.unsqueeze(1)), dim=0)
                else:
                    result_dict[target.item()] = w_batch.unsqueeze(1)  # ID: w_batch

        elif self.optimize_method == "CMA":
            parametrization = ng.p.Array(init=init_w[rank].cuda())
            target = torch.tensor(target).cuda()
            self.optimizer = ng.optimizers.registry['CMA'](parametrization=parametrization,
                                                           budget=self.config.attack["CMA"][
                                                               "epochs"])

            pbar = tqdm(range(self.config.attack["CMA"]["epochs"])) if self.use_pbar else range(
                self.config.attack["CMA"]["epochs"])

            for r in pbar:
                ng_data = [self.optimizer.ask() for _ in range(self.config.attack["CMA"]["number_samples"])]
                loss = [self.evaluate_loss(w=ng_data[i].value, target=target, ng_data=ng_data)
                        for i in
                        range(self.config.attack["CMA"]["number_samples"])]

                best_z = self.optimizer.provide_recommendation().value
                best_loss = self.evaluate_loss(w=best_z, target=target, ng_data=ng_data)

                population_best = torch.from_numpy(best_z).unsqueeze(0)
                imgs = self.synthesize(population_best).detach()
                if self.clip:
                    imgs = self.clip_images(imgs)
                if self.transformations:
                    imgs = self.transformations(imgs)
                outputs = self.target(imgs)
                if int(r + 1) % 50 == 0:
                    with torch.no_grad():
                        confidence_vector = outputs.softmax(dim=1)
                        confidences = torch.gather(
                            confidence_vector, 1, target.unsqueeze(1))
                        mean_conf = confidences.mean().detach().cpu()
                    print(
                        "Process ID {}\t Round {}\t - Population best Loss {:.5}\t - Population Best Confidence {:.4}  \t".format(
                            rank, r + 1,
                            round(best_loss, 4),
                            mean_conf))
                for z, l in zip(ng_data, loss):
                    self.optimizer.tell(z, l)
                if self.use_pbar:
                    pbar.set_description("Average Loss {:.6}".format(np.mean(loss)))

            w_batch = []
            for w in ng_data:
                w_batch.append(torch.from_numpy(w.value).unsqueeze(0).cpu())
            w_batch = torch.cat(w_batch, dim=0)

            with lock:
                if target.item() in result_dict:
                    result_dict[target.item()] = torch.cat((result_dict[target.item()], w_batch.unsqueeze(1)), dim=0)
                else:
                    result_dict[target.item()] = w_batch.unsqueeze(1)  # ID: w_batch

    def synthesize(self, w):
        c = None
        w_expanded = self.mapping(w.cuda(), c, truncation_psi=0.5, truncation_cutoff=8)
        imgs = self.generator(w_expanded,
                              noise_mode='const',
                              force_fp32=True)
        return imgs


    def clip_images(self, imgs):
        lower_limit = torch.tensor(-1.0).float().to(imgs.device)
        upper_limit = torch.tensor(1.0).float().to(imgs.device)
        imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
        imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
        return imgs

    def evaluate_loss(self, w, target, ng_data=None):
        w = torch.Tensor(w).unsqueeze(0)
        CE_loss = torch.nn.CrossEntropyLoss()
        img = self.synthesize(w)
        if self.clip:
            img = self.clip_images(img)

        if self.transformations:
            img = self.transformations(img)

        output = self.target(img)
        # Confidence Matching Loss
        if self.config.attack["confidence_matching_loss"] == 'max_margin':
            loss_cls = max_margin_loss(output, target)
        elif self.config.attack["confidence_matching_loss"] == 'poincare':
            loss_cls = poincare_loss(output, target)
        elif self.config.attack["confidence_matching_loss"] == 'cross_entropy':
            loss_cls = CE_loss(output, target)
        else:
            raise NotImplementedError
        if ng_data is None:
            return loss_cls.item()
        loss = loss_cls
        return loss.item()
