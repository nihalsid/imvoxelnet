import mmcv
import torch
from mmdet3d.utils.distinct_colors import DistinctColors
from mmdet3d.utils.misc import write_bbox


def single_gpu_test(model, data_loader, show=False, out_dir=None):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    # distinct_colors = DistinctColors()
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            model.module.show_results(data, result, out_dir)
            # for each label find the best score index
            # assert len(result) == 1
            # best_score_index = {}
            # best_scores_for_label = {}
            # for label in torch.unique(result[0]['labels_3d']).cpu().tolist():
            #     best_scores_for_label[label] = 0
            #     best_score_index[label] = -1
            # for label in torch.unique(result[0]['labels_3d']).cpu().tolist():
            #     for idx in range(result[0]['scores_3d'].shape[0]):
            #         if result[0]['labels_3d'][idx] == label:
            #             if result[0]['scores_3d'][idx] > best_scores_for_label[label]:
            #                 best_scores_for_label[label] = result[0]['scores_3d'][idx]
            #                 best_score_index[label] = idx
            # print(best_scores_for_label)
            # print(best_score_index)
            # corners = result[0]['boxes_3d'].corners
            # print(corners.shape)
            # for label in best_score_index.keys():
            #     write_bbox(corners[best_score_index[label], :].numpy(), distinct_colors.get_color(label), f"testbbox_{label:02d}.obj")

            # create a bounding box for that label

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
