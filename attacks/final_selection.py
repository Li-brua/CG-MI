import torch
import torch.nn.functional as F
from attacks.utils.stylegan import create_image
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from memory_profiler import profile

def scores_by_transform(imgs,
                        targets,
                        target_model,
                        transforms,
                        iterations=100):

    score = torch.zeros_like(
        targets, dtype=torch.float32).to(imgs.device)

    with torch.no_grad():
        for i in range(iterations):
            imgs_transformed = transforms(imgs)
            prediction_vector = target_model(imgs_transformed).softmax(dim=1)
            score += torch.gather(prediction_vector, 1,
                                  targets.unsqueeze(1)).squeeze()
        score = score / iterations
    return score

def perform_final_selection(w, generator, mapping, config, targets, target_model, samples_per_target,
                            approach, iterations, batch_size, device):
    target_values = set(targets.cpu().tolist())
    final_targets = []
    final_w = []
    target_model.eval()
    torch.manual_seed(42)
    if approach.strip() == 'transforms':
        transforms = T.Compose([
            T.RandomResizedCrop(size=(224, 224),
                                scale=(0.5, 0.9),
                                ratio=(0.8, 1.2)),
            T.RandomHorizontalFlip(0.5)
        ])

    for step, target in enumerate(target_values):
        mask = torch.where(targets == target, True, False)
        w_masked = w[mask]
        # candidates = create_image(w_masked,
        #                           generator,
        #                           mapping,
        #                           crop_size=config.attack_center_crop,
        #                           resize=config.attack_resize,
        #                           device=device).cpu()
        candidates = []
        targets_masked = []
        for i in range(0, len(w_masked), batch_size):
            batch_w_masked = w_masked[i:i + batch_size]
            batch_candidates = create_image(batch_w_masked,
                                            generator,
                                            mapping,
                                            crop_size=config.attack_center_crop,
                                            resize=config.attack_resize,
                                            device=device).cpu()
            batch_targets_masked = targets[mask][i:i + batch_size].cpu()

            candidates.append(batch_candidates)
            targets_masked.append(batch_targets_masked)
        candidates = torch.cat(candidates, dim=0)

        targets_masked = targets[mask].cpu()
        scores = []
        dataset = TensorDataset(candidates, targets_masked)
        for imgs, t in DataLoader(dataset, batch_size=batch_size):
            imgs, t = imgs.to(device), t.to(device)

            scores.append(scores_by_transform(imgs,
                                                  t,
                                                  target_model,
                                                  transforms,
                                                  iterations))
            # torch.cuda.memory_allocated()
        scores = torch.cat(scores, dim=0).cpu()
        indices = torch.sort(scores, descending=True).indices
        # print(scores[indices[0].item()])
        selected_indices = indices[:samples_per_target]
        final_targets.append(targets_masked[selected_indices].cpu())
        final_w.append(w_masked[selected_indices].cpu())

        # if rtpt:
        #     rtpt.step(
        #         subtitle=f'Sample Selection step {step} of {len(target_values)}')
    final_targets = torch.cat(final_targets, dim=0)
    final_w = torch.cat(final_w, dim=0)
    return final_w, final_targets
