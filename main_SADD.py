''' README FOR THIS FILE

NETWORK - 1 : FROZE AASIST AND PROCESS FRAME WISE LPRN without batch normalization'''

import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from LPRN_model import LPRN_SADD
from data_utils import (Dataset_ASVspoof2019_train_SADD,
                        Dataset_ASVspoof2019_dev_SADD, genSpoof_list_spk)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])

    ## made changes here for non target speaker removal
    dev_trial_path = '/home/s22004/research/aasist_spk_aware_tomi/cm_files/dev_CM_ASV2019.txt'
    eval_trial_path = '/home/s22004/research/aasist_spk_aware_tomi/cm_files/eval_CM_ASV2019.txt'

    # define model related paths
    model_tag = "LPC_aasist_LPRN_SADD_batch_24"
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"

    eval_score_path = model_tag / config["eval_output"]
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = torch.device('cuda:5')
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)
    ## load saved weights from AASIST
    saved_weights = torch.load('/your_path_to_AASIST/AASIST.pth')
    model.load_state_dict(saved_weights)
    # Set requires_grad to False for all parameters in AASIST
    for param in model.parameters():
        param.requires_grad = False
    print("AASIST model loaded")

    # define model architecture - LPRN
    model2 = LPRN_SADD().to(device)

    ## define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        print("checking")
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        model2.load_state_dict(
            torch.load(config["model_path2"], map_location=device))
        print("AASIST loaded : {}".format(config["model_path"]))
        print("LPRN SADD loaded : {}".format(config["model_path2"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, model2, device,
                                eval_score_path, eval_trial_path)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path /
                           config["asv_score_path"],
                           output_file="your_path.txt")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model2.parameters(), optim_config) ## FOR LPRN
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    print("begin training")

    for epoch in range(config["num_epochs"]):
    
        print("Start training epoch {:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, model2, 
        optimizer, device, scheduler, config)
        print("loss", running_loss)

        ## !!!!CAUTION !!!!need to save only LPRN model, as AASIST is frozen. during eval just take frozen AASIST and saved "MODEL 2 aka LPRN"
        torch.save(model2.state_dict(),
                       model_save_path / "epoch_{}_loss_{:.5f}.pth".format(epoch,running_loss))
        produce_evaluation_file(dev_loader, model, model2, device,
                                metric_path/"dev_score.txt", dev_trial_path)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            asv_score_file=database_path/config["asv_score_path"],
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
            running_loss, dev_eer, dev_tdcf))
        ## write in a log file
        with open("/log/train_log", "a") as f1:
            f1.write(("\n EPOCH {:03d}".format(epoch)))
            f1.write("\nTrain Loss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
            running_loss, dev_eer, dev_tdcf))

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print(f"best model find at epoch: {epoch} and dev eer is {dev_eer}")
            best_dev_eer = dev_eer
            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, model2,device,
                                        eval_score_path, eval_trial_path)
                eval_eer, eval_tdcf = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path / config["asv_score_path"],
                    output_file=metric_path /
                    "t-DCF_EER_{:03d}epo.txt".format(epoch))

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    log_text += "best tdcf, {:.4f}".format(eval_tdcf)
                    best_eval_tdcf = eval_tdcf
                    torch.save(model2.state_dict(),
                               model_save_path / "best.pth")
                if len(log_text) > 0:
                    print(log_text)

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        print(f"best dev eer: {best_dev_eer} at epoch: {epoch}")
        print(f"best dev tdcf: {best_dev_tdcf} at epoch: {epoch}")

    print(f"n_swa_update val: {n_swa_update}")
    print("AT LAST")
    print("------------------------------Start final evaluation ----------------------------------------")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, model2, device, eval_score_path,
                            eval_trial_path)
    eval_eer, eval_tdcf = calculate_tDCF_EER(cm_scores_file=eval_score_path,
                                             asv_score_file=database_path /
                                             config["asv_score_path"],
                                             output_file=model_tag / "t-DCF_EER.txt")

    print("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))

    torch.save(model2.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model2.state_dict(),
                   model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
        best_eval_eer, best_eval_tdcf))



def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    trn_list_path = (database_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))

    ## made changes here for non target speaker removal
    dev_trial_path = '/path_to_spkaware_protocol/dev_CM_ASV2019.txt'
    eval_trial_path = '/path_to_spkaware_protocol/eval_CM_ASV2019.txt'

    d_label_trn, file_train, spk_ids_train = genSpoof_list_spk(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train_SADD(list_IDs=file_train,
                                           labels=d_label_trn, spk_IDs = spk_ids_train,
                                           base_dir=trn_database_path,set_type="train")
    gen = torch.Generator()
    gen.manual_seed(seed)

    print(f"train_set view:> {train_set[46]}")
    # # num_samples = 2000
    # # subset_indices = random.sample(range(len(train_set)), num_samples)
    # # subset_train_set = torch.utils.data.Subset(train_set, subset_indices)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=False,
                            worker_init_fn=seed_worker,
                            generator=gen)

    d_label_dev, file_dev, spk_ids_dev = genSpoof_list_spk(dir_meta=dev_trial_path,
                                is_train=True,is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_dev_SADD(list_IDs=file_dev,
                                           labels=d_label_dev, spk_IDs = spk_ids_dev,
                                           base_dir=dev_database_path,set_type="dev")
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False)

    d_label_eval, file_eval, spk_ids_eval = genSpoof_list_spk(dir_meta=eval_trial_path,
                              is_train=True,is_eval=False)

    eval_set = Dataset_ASVspoof2019_dev_SADD(list_IDs=file_eval,
                                           labels=d_label_eval, spk_IDs = spk_ids_eval,
                                           base_dir=eval_database_path,set_type="eval")
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=False)
    print(f"eval_set view:> {eval_set[1]}")
    print(f"eval_set view:> {eval_set[41]}")

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    model2,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    model2.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id, batch_LPC_res in data_loader:
        batch_x = batch_x.to(device)
        batch_LPC_res = batch_LPC_res.to(device)
        batch_LPC_res = batch_LPC_res.unsqueeze(1)
        with torch.no_grad():
            batch_hidden, _ = model(batch_x)
            batch_out = model2(batch_LPC_res, batch_hidden)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        
        # add outputs

        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            # assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    model2,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    running_loss = 0
    num_total = 0.0
    model.eval()
    model2.train()
    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y, batch_LPC_res in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_LPC_res = batch_LPC_res.to(device)
        batch_LPC_res = batch_LPC_res.unsqueeze(1)

        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        with torch.no_grad():
            batch_hidden, _ = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_out = model2(batch_LPC_res, batch_hidden)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Aware Deepfake Detectors")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        default="./config/AASIST-N1.conf")
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="SADD",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    main(parser.parse_args())
