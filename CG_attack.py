import argparse
import random
import csv

import numpy as np
import math
from torch.utils.data import TensorDataset
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1
import traceback
import os  # a
import torch.multiprocessing
from pathlib import Path

os.environ["WANDB_API_KEY"] = 'resnet18_celeba'
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from attacks.final_selection import perform_final_selection
from attacks.Gradient_free_optimize import Optimization
from attacks.metrics.fid_score import FID_Score
from attacks.metrics.prcd import PRCD
from attacks.metrics.classification_acc import ClassificationAccuracy
from attacks.datasets.custom_subset import ClassSubset
from attacks.utils.attack_config_parser import AttackConfigParser
from attacks.utils.datasets import get_facescrub_idx_to_class, get_stanford_dogs_idx_to_class, create_target_dataset
from attacks.utils.stylegan import create_image, load_generator
from attacks.utils.wandb import *


def main():
    ####################################
    #        Attack Preparation        #
    ####################################

    # Set devices
    torch.cuda.device_count()
    torch.set_num_threads(24)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    # Set seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load idx to class mappings
    idx_to_class = None
    if config.dataset.lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif config.dataset.lower() == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:
        class KeyDict(dict):
            def __missing__(self, key):
                return key

        idx_to_class = KeyDict()

    # Load pre-trained StyleGan2 components
    G = load_generator(config.stylegan_model)

    # Load target model and set dataset
    target_model = config.create_target_model()
    target_model_name = target_model.name
    target_dataset = config.get_target_dataset()

    # Distribute models
    target_model = torch.nn.DataParallel(
        target_model, device_ids=gpu_devices)
    target_model.name = target_model_name
    synthesis = G.synthesis.cuda()
    mapping = G.mapping.cuda()

    # Create target vector
    targets = config.create_target_vector()

    # Initialize wandb logging
    if config.logging:
        # optimizer = config.create_optimizer(params=[w])
        wandb_run = init_wandb_logging(
            target_model_name, config, args)
        run_id = wandb_run.id

    # Print attack configuration
    print(
        f'Start attack against {target_model.name}',
        f'and targets {targets.cpu().numpy()}.'
    )
    print(f'\nAttack parameters')
    for key in config.attack:
        print(f'\t{key}: {config.attack[key]}')
    print(
        f'Performing attack on {torch.cuda.device_count()} gpus and an effective optimize method of {config.attack["optimizer"]}.')

    # Initialize RTPT
    rtpt = None

    # Create attack transformations
    attack_transformations = config.create_attack_transformations()

    # Create initial vectors
    targets_init = torch.repeat_interleave(torch.unique(targets),
                                           config.attack["CMA"]["iters"])
    w = create_initial_vectors(
        config, G, target_model, targets_init)

    if config.logging:
        Path("results").mkdir(parents=True, exist_ok=True)
        init_w_path = f"results/init_w_{run_id}.pt"
        torch.save(w.detach(), init_w_path)
        wandb.save(init_w_path)


    ####################################
    #         Attack Iteration         #
    ####################################
    optimization = Optimization(
        target_model, synthesis, mapping, G.z_dim, attack_transformations, config, device)
    # Collect results
    mean_time = []
    w_unfilter = {}
    lock = torch.multiprocessing.Manager().Lock()
    if config.attack['optimizer'] == 'CMA':
        targets_unfilter = torch.repeat_interleave(torch.unique(targets),
                                                   config.attack["CMA"]["iters"])
    elif config.attack['optimizer'] == 'BO':
        targets_unfilter = torch.repeat_interleave(torch.unique(targets),
                                                   config.attack["BO"]["iters"])

    # Prepare Multi Process for attack
    for i in range(math.ceil(len(targets_unfilter) / config.attack["num_workers"])):
        targets_batch = targets_unfilter[i * int(config.attack["num_workers"]):(i + 1) * int(
            config.attack["num_workers"])].cpu()
        print("Start Attack Identity:{} ".format(torch.unique(targets_batch).tolist()))
        torch.cuda.empty_cache()
        start_time = time.perf_counter()

        if config.attack['optimizer'] == 'BO':
            ctx = torch.multiprocessing.get_context("spawn")
            if len(targets_batch) < config.attack["num_workers"]:
                epochs = len(targets_batch)
                pool = ctx.Pool(processes=len(targets_batch))
            else:
                epochs = config.attack["num_workers"]
                pool = ctx.Pool(processes=config.attack["num_workers"])
            manager = ctx.Manager()

            result_dict = manager.dict()

            for rank in range(epochs):
                p = pool.apply_async(optimization.optimize,
                                     args=(torch.unique(targets_batch), None, result_dict, rank, lock),
                                     error_callback=lambda e: print(e))
            pool.close()
            pool.join()

            for key, result in result_dict.items():
                if key in w_unfilter:
                    w_unfilter[key] = torch.cat((w_unfilter[key], result), dim=0)
                else:
                    w_unfilter[key] = result

        elif config.attack['optimizer'] == 'CMA':
            ctx = torch.multiprocessing.get_context("spawn")
            if len(targets_batch) < config.attack["num_workers"]:
                epochs = len(targets_batch)
                pool = ctx.Pool(processes=len(targets_batch))
            else:
                epochs = config.attack["num_workers"]
                pool = ctx.Pool(processes=config.attack["num_workers"])
            manager = ctx.Manager()
            # print(init_w.shape)
            # init_w = torch.randn([config.attack["CMA"]["iters"], G.z_dim]).cuda()
            init_w = w[i * int(config.attack["num_workers"]):(i + 1) * int(
                config.attack["num_workers"])].cpu()
            result_dict = manager.dict()

            for rank in range(epochs):
                # print(f"Current working directory in child process: {os.getcwd()}")
                p = pool.apply_async(optimization.optimize, args=([targets_batch[rank]], init_w, result_dict, rank, lock),
                                     error_callback=lambda e: print(e))
            pool.close()
            pool.join()

            for key, result in result_dict.items():
                if key in w_unfilter:
                    w_unfilter[key] = torch.cat((w_unfilter[key], result), dim=0)
                else:
                    w_unfilter[key] = result

        end_time = time.perf_counter()
        mean_time.append(end_time - start_time)

    print(f"Attack Mean Time per ID {int(np.mean(mean_time) / 60)} min")

    w_optimized = []
    w_unfilter = sorted(w_unfilter.items(), key=lambda x: x[0])

    for t, w_batch in w_unfilter:
        w_optimized.append(w_batch)

    w_optimized = torch.cat(w_optimized, dim=0)


    if config.logging:
        optimized_w_path = f"results/optimized_z_{run_id}.pt"
        torch.save(w_optimized.detach(), optimized_w_path)
        wandb.save(optimized_w_path)


    evaluation_model = config.create_evaluation_model()
    evaluation_model = torch.nn.DataParallel(evaluation_model)
    evaluation_model.to(device)
    evaluation_model.eval()

    ###################
    targets_unselect = torch.repeat_interleave(torch.unique(targets),
                                               config.attack["CMA"]["iters"] * config.attack["CMA"][
                                                   "number_samples"])
    class_acc_evaluator = ClassificationAccuracy(
        evaluation_model, device=device)
    acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
        w_optimized, targets_unselect, synthesis, mapping, config, batch_size=40, resize=299, rtpt=rtpt)

    if config.logging:
        try:
            filename_precision = write_precision_list(
                f'results/precision_list_unfiltered_{run_id}', precision_list)
            wandb.save(filename_precision)
        except:
            pass

    print(
        f'\nUnfilter Evaluation of {w_optimized.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}',
        f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
    )

    w_best, _ = perform_final_selection(w_optimized, synthesis, mapping, config,
                                        targets_unselect, target_model, device='cuda', batch_size=10,
                                        samples_per_target=config.attack["CMA"]["num_select"],
                                        approach='transforms', iterations=100)

    targets_best = torch.repeat_interleave(torch.unique(targets), config.attack["CMA"]["num_select"])
    acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
        w_best, targets_best, synthesis, mapping, config, batch_size=40, resize=299, rtpt=rtpt)

    if config.logging:
        filename_precision = write_precision_list(
            f'results/precision_list_filtered_{run_id}', precision_list)
        wandb.save(filename_precision)

    print(
        f'\nfilter Evaluation of {w_best.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}',
        f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
    )

    if config.logging:
        optimized_selected_z_path = f"results/optimized_selected_z_{run_id}.pt"
        torch.save(w_best.detach(), optimized_selected_z_path)
        wandb.save(optimized_selected_z_path)

    final_w = w_best
    final_targets = targets.cpu()
    torch.cuda.empty_cache()

    del target_model

    ####################################
    #      Final Attack Accuracy       #
    ####################################

    # Compute attack accuracy with evaluation model on all generated samples
    try:
        evaluation_model = config.create_evaluation_model()
        evaluation_model = torch.nn.DataParallel(evaluation_model)
        evaluation_model.to(device)
        evaluation_model.eval()
        class_acc_evaluator = ClassificationAccuracy(
            evaluation_model, device=device)
        acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
            final_w, targets, synthesis, mapping, config, batch_size=40, resize=299,
            rtpt=rtpt)

        if config.logging:
            try:
                filename_precision = write_precision_list(
                    f'results/precision_list_{run_id}', precision_list)
                wandb.save(filename_precision)
            except:
                pass
        print(
            f'\nEvaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}',
            f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
        )
        del evaluation_model

    except Exception:
        print(traceback.format_exc())

    ####################################
    #    FID Score and GAN Metrics     #
    ####################################

    fid_score = None
    precision, recall = None, None
    density, coverage = None, None
    try:
        # set transformations
        crop_size = config.attack_center_crop
        target_transform = T.Compose(
            [T.ToTensor(), T.Resize((299, 299)), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # create datasets
        attack_dataset = TensorDataset(
            final_w, final_targets)
        attack_dataset.targets = final_targets
        training_dataset = create_target_dataset(
            target_dataset, target_transform)
        training_dataset = ClassSubset(
            training_dataset, target_classes=torch.unique(final_targets).cpu().tolist())

        # compute FID score
        fid_evaluation = FID_Score(
            training_dataset, attack_dataset, device=device, crop_size=crop_size, generator=synthesis, mapping=mapping,
            batch_size=40, dims=2048, num_workers=8, gpu_devices=gpu_devices)
        fid_score = fid_evaluation.compute_fid(rtpt)
        print(
            f'FID score computed on {final_w.shape[0]} attack samples and {config.dataset}: {fid_score:.4f}'
        )

        # compute precision, recall, density, coverage
        prdc = PRCD(training_dataset, attack_dataset, device=device, crop_size=crop_size, generator=synthesis,
                    mapping=mapping, batch_size=40, dims=2048, num_workers=8, gpu_devices=gpu_devices)
        precision, recall, density, coverage = prdc.compute_metric(num_classes=config.num_classes, k=3, rtpt=rtpt)
        print(
            f' Precision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
        )

    except Exception:
        print(traceback.format_exc())

    ####################################
    #         Feature Distance         #
    ####################################
    avg_dist_inception = None
    avg_dist_facenet = None
    try:
        # Load Inception-v3 evaluation model and remove final layer
        evaluation_model_dist = config.create_evaluation_model()
        evaluation_model_dist.model.fc = torch.nn.Sequential()
        evaluation_model_dist = torch.nn.DataParallel(evaluation_model_dist, device_ids=gpu_devices)
        evaluation_model_dist.to(device)
        evaluation_model_dist.eval()

        # Compute average feature distance on Inception-v3
        evaluate_inception = DistanceEvaluation(
            evaluation_model_dist, synthesis, mapping, 299, config.attack_center_crop, target_dataset, config.seed)
        avg_dist_inception, mean_distances_list = evaluate_inception.compute_dist(
            final_w, final_targets, batch_size=40, rtpt=rtpt)

        if config.logging:
            try:
                filename_distance = write_precision_list(
                    f'results/distance_inceptionv3_list_filtered_{run_id}', mean_distances_list)
                wandb.save(filename_distance)
            except:
                pass

        print('Mean Distance on Inception-v3: ', avg_dist_inception.cpu().item())
        # Compute feature distance only for facial images
        if target_dataset in ['facescrub', 'celeba_identities', 'celeba_attributes']:
            # Load FaceNet model for face recognition
            facenet = InceptionResnetV1(pretrained='vggface2')
            facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
            facenet.to(device)
            facenet.eval()

            # Compute average feature distance on facenet
            evaluater_facenet = DistanceEvaluation(
                facenet, synthesis, mapping, 160, config.attack_center_crop, target_dataset, config.seed)
            avg_dist_facenet, mean_distances_list = evaluater_facenet.compute_dist(
                final_w, final_targets, batch_size=40, rtpt=rtpt)
            if config.logging:
                filename_distance = write_precision_list(
                    f'results/distance_facenet_list_filtered_{run_id}', mean_distances_list)
                wandb.save(filename_distance)

            print('Mean Distance on FaceNet: ', avg_dist_facenet.cpu().item())
    except Exception:
        print(traceback.format_exc())

    ####################################
    #          Finish Logging          #
    ####################################

    if rtpt:
        rtpt.step(subtitle=f'Finishing up')

    # Logging of final results
    if config.logging:
        print('Finishing attack, logging results and creating sample images.')
        num_classes = int(len(targets) / config.attack["CMA"]["num_select"])
        num_imgs = 1
        # Sample final images from the first and last classes
        label_subset = set(list(set(targets.tolist()))[
                           :int(num_classes / 2)] + list(set(targets.tolist()))[-int(num_classes / 2):])
        log_imgs = []
        log_targets = []
        log_predictions = []
        log_max_confidences = []
        log_target_confidences = []
        # Log images with smallest feature distance
        for label in label_subset:
            mask = torch.where(final_targets == label, True, False)
            w_masked = final_w[mask][:num_imgs].cuda()
            imgs = create_image(
                w_masked, synthesis, mapping, crop_size=config.attack_center_crop, resize=config.attack_resize)
            log_imgs.append(imgs)
            log_targets += [label for i in range(num_imgs)]
            log_predictions.append(torch.tensor(predictions)[mask][:num_imgs])
            log_max_confidences.append(
                torch.tensor(maximum_confidences)[mask][:num_imgs])
            log_target_confidences.append(
                torch.tensor(target_confidences)[mask][:num_imgs])

        log_imgs = torch.cat(log_imgs, dim=0)
        log_predictions = torch.cat(log_predictions, dim=0)
        log_max_confidences = torch.cat(log_max_confidences, dim=0)
        log_target_confidences = torch.cat(log_target_confidences, dim=0)

        log_final_images(log_imgs, log_predictions, log_max_confidences,
                         log_target_confidences, idx_to_class)

        # Find closest training samples to final results
        log_nearest_neighbors(log_imgs, log_targets, evaluation_model_dist,
                              'InceptionV3', target_dataset, img_size=299, seed=config.seed)

        # Use FaceNet only for facial images
        facenet = InceptionResnetV1(pretrained='vggface2')
        facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
        facenet.to(device)
        facenet.eval()
        if target_dataset in ['facescrub', 'celeba_identities', 'celeba_attributes']:
            log_nearest_neighbors(log_imgs, log_targets, facenet, 'FaceNet',
                                  target_dataset, img_size=160, seed=config.seed)

        # Final logging
        final_wandb_logging(avg_correct_conf, avg_total_conf, acc_top1, acc_top5,
                            avg_dist_facenet, avg_dist_inception, fid_score, precision, recall, density, coverage)


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--no_rtpt',
                        action='store_false',
                        dest="rtpt",
                        help='Disable RTPT')
    return parser


def parse_arguments(parser):
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load attack config
    config = AttackConfigParser(args.config)
    return config, args


def create_initial_vectors(config, G, target_model, targets):
    with torch.no_grad():
        w = config.create_candidates(G, target_model, targets)
    return w


def write_precision_list(filename, precision_list):
    filename = f"{filename}.csv"
    with open(filename, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for row in precision_list:
            wr.writerow(row)
    return filename


if __name__ == '__main__':
    main()
