"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import pdb
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from torchvision.models import resnet18
import numpy as np
from analysis import Logger, AverageMeter, accuracy, binary_accuracy

from CUB import probe, tti, gen_cub_synthetic, hyperopt
from CUB.dataset import load_data, find_class_imbalance
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from CUB.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY
import CUB.resnet_SD as models
from CUB.utils import kd_loss_function, feature_loss_function



def run_epoch_simple(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            #inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = torch.autograd.Variable(inputs).cuda()
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda()
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var
        
        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters
    return loss_meter, acc_meter

def run_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, batch_time, data_time, losses_m, top1, middle1_losses, middle2_losses, middle3_losses, losses1_kd, losses2_kd, losses3_kd, feature_losses_1, feature_losses_2, feature_losses_3, total_losses, middle1_top1, middle2_top1, middle3_top1, is_training):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()#.float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var

        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var


        outputs, middle_output1, middle_output2, middle_output3, final_fea, middle1_fea, middle2_fea, middle3_fea = model(inputs.cuda())
        

        losses = []
        l_middle1_loss = []
        l_middle2_loss = []
        l_middle3_loss = []
        l_temp4 = []
        l_loss1by4 = []
        l_loss2by4 = []
        l_loss3by4 = []
        out_start = 0


        if not args.bottleneck:
            loss_main = criterion(outputs[0], labels_var)


            losses.append(loss_main)
            l_middle1_loss.append(criterion(middle_output1[0], labels_var))
            l_middle2_loss.append(criterion(middle_output2[0], labels_var))
            l_middle3_loss.append(criterion(middle_output3[0], labels_var))
            out_start = 1
        if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
            for i in range(len(attr_criterion)):
                losses.append(args.attr_loss_weight * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]))
                l_middle1_loss.append(args.attr_loss_weight * attr_criterion[i](middle_output1[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]))
                l_middle2_loss.append(args.attr_loss_weight * attr_criterion[i](middle_output2[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]))
                l_middle3_loss.append(args.attr_loss_weight * attr_criterion[i](middle_output3[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]))
                l_temp4.append(outputs[i+out_start] / args.temperature)


        if args.bottleneck: #attribute accuracy
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))

            
            sigmoid_middle_output1 = torch.nn.Sigmoid()(torch.cat(middle_output1, dim=1))
            sigmoid_middle_output2 = torch.nn.Sigmoid()(torch.cat(middle_output2, dim=1))
            sigmoid_middle_output3 = torch.nn.Sigmoid()(torch.cat(middle_output3, dim=1))

            temp4 = sigmoid_outputs/args.temperature            
            loss1by4 = kd_loss_function(sigmoid_middle_output1, temp4.detach(), args) * (args.temperature**2)
            loss2by4 = kd_loss_function(sigmoid_middle_output2, temp4.detach(), args) * (args.temperature**2)
            loss3by4 = kd_loss_function(sigmoid_middle_output3, temp4.detach(), args) * (args.temperature**2)

            middle1_prec1 = binary_accuracy(sigmoid_middle_output1, attr_labels)
            middle2_prec1 = binary_accuracy(sigmoid_middle_output2, attr_labels)
            middle3_prec1 = binary_accuracy(sigmoid_middle_output3, attr_labels)
            
            losses1_kd.update(loss1by4, inputs.size(0))            
            losses2_kd.update(loss2by4, inputs.size(0))            
            losses3_kd.update(loss3by4, inputs.size(0))

            top1.update(acc.data.cpu().numpy(), inputs.size(0))
            middle1_top1.update(middle1_prec1.data.cpu().numpy(), inputs.size(0))
            middle2_top1.update(middle2_prec1.data.cpu().numpy(), inputs.size(0))
            middle3_top1.update(middle3_prec1.data.cpu().numpy(), inputs.size(0))



        else:
            acc = accuracy(outputs[0], labels, topk=(1,)) #only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        if attr_criterion is not None:
            if args.bottleneck:
                ototal_loss = sum(losses)/ args.n_attributes
                middle1_loss = sum(l_middle1_loss)/ args.n_attributes
                middle2_loss = sum(l_middle2_loss)/ args.n_attributes
                middle3_loss = sum(l_middle3_loss)/ args.n_attributes
                temp4 = sum(l_temp4) / args.n_attributes
                temp4 = torch.softmax(temp4, dim=1)
                loss1by4 = sum(l_loss1by4)/ args.n_attributes
                loss2by4 = sum(l_loss2by4)/ args.n_attributes
                loss3by4 = sum(l_loss3by4)/ args.n_attributes

                
            else: #cotraining, loss by class prediction and loss by attribute prediction have the same weight
                ototal_loss = losses[0] + sum(losses[1:])
                middle1_loss = l_middle1_loss[0] + sum(l_middle1_loss[1:])
                middle2_loss = l_middle2_loss[0] + sum(l_middle2_loss[1:])
                middle3_loss = l_middle3_loss[0] + sum(l_middle3_loss[1:])
                temp4 = sum(l_temp4) / args.n_attributes
                temp4 = torch.softmax(temp4, dim=1)
                loss1by4 = l_loss1by4[0] + sum(l_loss1by4[1:])
                loss2by4 = l_loss2by4[0] + sum(l_loss2by4[1:])
                loss3by4 = l_loss3by4[0] + sum(l_loss3by4[1:])
                if args.normalize_loss:
                    ototal_loss = ototal_loss / (1 + args.attr_loss_weight * args.n_attributes)
                    middle1_loss = middle1_loss / (1 + args.attr_loss_weight * args.n_attributes)
                    middle2_loss = middle2_loss / (1 + args.attr_loss_weight * args.n_attributes)
                    middle3_loss = middle3_loss / (1 + args.attr_loss_weight * args.n_attributes)
                    loss1by4 = loss1by4 / (1 + args.attr_loss_weight * args.n_attributes)
                    loss2by4 = loss2by4 / (1 + args.attr_loss_weight * args.n_attributes)
                    loss3by4 = loss3by4 / (1 + args.attr_loss_weight * args.n_attributes)
        
        
        else: #finetune
            ototal_loss = sum(losses)
            middle1_loss = sum(l_middle1_loss)
            middle2_loss = sum(l_middle2_loss)
            middle3_loss = sum(l_middle3_loss)
            temp4 = sum(l_temp4)
            temp4 = torch.softmax(temp4, dim=1)    
            loss1by4 = sum(l_loss1by4)
            loss2by4 = sum(l_loss2by4)
            loss3by4 = sum(l_loss3by4)

        middle1_losses.update(middle1_loss.item(), inputs.size(0))
        middle2_losses.update(middle2_loss.item(), inputs.size(0))
        middle3_losses.update(middle3_loss.item(), inputs.size(0))
            

        feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) 
        feature_losses_1.update(feature_loss_1, inputs.size(0))
        feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) 
        feature_losses_2.update(feature_loss_2, inputs.size(0))
        feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach()) 
        feature_losses_3.update(feature_loss_3, inputs.size(0))

        total_loss = (1 - args.alpha) * (ototal_loss + middle1_loss + middle2_loss + middle3_loss) + \
                    args.alpha * (loss1by4 + loss2by4 + loss3by4) + \
                    args.beta * (feature_loss_1 + feature_loss_2 + feature_loss_3)
        

        
        total_losses.update(total_loss.item(), inputs.size(0))
        






        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    
    return loss_meter, acc_meter, top1, middle1_losses, middle2_losses, middle3_losses, losses1_kd, losses2_kd, losses3_kd, feature_losses_1, feature_losses_2, feature_losses_3, total_losses, middle1_top1, middle2_top1, middle3_top1




def train(model, args):
    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)

    if os.path.exists(args.log_dir): # job restarted by cluster
        for f in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, f))
    else:
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()
    model = models.__dict__['multi_resnet18_kd'](num_classes=N_CLASSES, n_attributes=args.n_attributes, bottleneck=True, args = args).cuda()
    #model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_attr and not args.no_img:
        attr_criterion = [] #separate criterion (loss function) for each attribute
        if args.weighted_loss:
            assert(imbalance is not None)
            for ratio in imbalance:
                attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda()))
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001, min_lr=0.00001, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    if args.ckpt: #retraining
        train_loader = load_data([train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = None
    else:
        train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr)



    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        middle1_losses = AverageMeter()
        middle2_losses = AverageMeter()
        middle3_losses = AverageMeter()
        losses1_kd = AverageMeter()
        losses2_kd = AverageMeter()
        losses3_kd = AverageMeter()
        feature_losses_1 = AverageMeter()
        feature_losses_2 = AverageMeter()
        feature_losses_3 = AverageMeter()
        total_losses = AverageMeter()
        middle1_top1 = AverageMeter()
        middle2_top1 = AverageMeter()
        middle3_top1 = AverageMeter()
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, args, is_training=True)
        else:
            train_loss_meter, train_acc_meter,top1, middle1_losses, middle2_losses, middle3_losses, losses1_kd, losses2_kd, losses3_kd, feature_losses_1, feature_losses_2, feature_losses_3, total_losses, middle1_top1, middle2_top1, middle3_top1 = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, batch_time, 
                                                            data_time, losses, top1, middle1_losses, middle2_losses, middle3_losses, losses1_kd, losses2_kd, losses3_kd, 
                                                            feature_losses_1, feature_losses_2, feature_losses_3, total_losses, middle1_top1, middle2_top1, middle3_top1, is_training=True,)
 
        if not args.ckpt: # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()


            val_batch_time = AverageMeter()
            val_data_time = AverageMeter()
            val_losses = AverageMeter()
            val_top1 = AverageMeter()
            val_middle1_losses = AverageMeter()
            val_middle2_losses = AverageMeter()
            val_middle3_losses = AverageMeter()
            val_losses1_kd = AverageMeter()
            val_losses2_kd = AverageMeter()
            val_losses3_kd = AverageMeter()
            val_feature_losses_1 = AverageMeter()
            val_feature_losses_2 = AverageMeter()
            val_feature_losses_3 = AverageMeter()
            val_total_losses = AverageMeter()
            val_middle1_top1 = AverageMeter()
            val_middle2_top1 = AverageMeter()
            val_middle3_top1 = AverageMeter()            
        
            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, args, is_training=False)
                else:
                    val_loss_meter, val_acc_meter, val_top1, val_middle1_losses, val_middle2_losses, val_middle3_losses, val_losses1_kd, val_losses2_kd, val_losses3_kd, val_feature_losses_1, val_feature_losses_2, val_feature_losses_3, val_total_losses, val_middle1_top1, val_middle2_top1, val_middle3_top1 = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, attr_criterion, args,  val_batch_time, 
                                                            val_data_time, val_losses, val_top1, val_middle1_losses, val_middle2_losses, val_middle3_losses, val_losses1_kd, val_losses2_kd, val_losses3_kd, 
                                                            val_feature_losses_1, val_feature_losses_2, val_feature_losses_3, val_total_losses, val_middle1_top1, val_middle2_top1, val_middle3_top1, is_training=False)

        else: #retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pth' % args.seed))
            #if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
            #    break

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                'Val loss: %.4f\tVal acc: %.4f\t'
                'Best val epoch: %d\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                'Val loss: %.4f\tVal acc: %.4f\t'
                'Best val epoch: %d\n'
                'train middle loss 1: %d\n'
                'train middle loss 2: %d\n'
                'train middle loss 3: %d\n'
                'val middle loss 1: %d\n'
                'val middle loss 2: %d\n'
                'val middle loss 3: %d\t'                                                 
                'train middle acc 1: %d\n'
                'train middle acc 2: %d\n'
                'train middle acc 3: %d\n'
                'val middle acc 1: %d\n'
                'val middle acc 2: %d\n'
                'val middle acc 3: %d\t'                                                                    
                'train kd loss 1: %d\n'
                'train kd loss 2: %d\n'
                'train kd loss 3: %d\n'
                'val kd loss 1: %d\n'
                'val kd loss 2: %d\n'
                'val kd loss 3: %d\t'                
                'train feature loss 1: %d\n'
                'train feature loss 2: %d\n'
                'train feature loss 3: %d\n'
                'val feature loss 1: %d\n'
                'val feature loss 2: %d\n'
                'val feature loss 3: %d\t'                                                                  
                % (epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg, best_val_epoch,
                middle1_losses.avg, middle2_losses.avg, middle3_losses.avg, val_middle1_losses.avg, val_middle2_losses.avg, val_middle3_losses.avg,
                middle1_top1.avg, middle2_top1.avg, middle3_top1.avg, val_middle1_top1.avg, val_middle2_top1.avg, val_middle3_top1.avg,
                losses1_kd.avg, losses2_kd.avg, losses3_kd.avg, val_losses1_kd.avg, val_losses2_kd.avg, val_losses3_kd.avg,
                feature_losses_1.avg, feature_losses_2.avg, feature_losses_3.avg, val_feature_losses_1.avg, val_feature_losses_2.avg, val_feature_losses_3.avg
                )) 
        logger.flush()
        
        if epoch <= stop_epoch:
            scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_lr())

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break
    
    
    test_loss_meter = AverageMeter()
    test_acc_meter = AverageMeter()
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')
    test_loader = load_data([test_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr)

    test_loss_meter, test_acc_meter = run_epoch(model, optimizer, test_loader, test_loss_meter, test_acc_meter, criterion, attr_criterion, args, is_training=False)

    print('Test loss: %.4f\tTest accuracy: %.4f\t'
                % (test_loss_meter.avg, test_acc_meter.avg))

def train_X_to_C(args):
    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                      n_attributes=args.n_attributes, expand_dim=args.expand_dim, three_class=args.three_class)
    train(model, args)

def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                            num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                                 num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_X_to_C_to_y(args):
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    train(model, args)

def train_X_to_y(args):
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux)
    train(model, args)

def train_X_to_Cy(args):
    model = ModelXtoCY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                       n_attributes=args.n_attributes, three_class=args.three_class, connect_CY=args.connect_CY)
    train(model, args)

def train_probe(args):
    probe.run(args)

def test_time_intervention(args):
    tti.run(args)

def robustness(args):
    gen_cub_synthetic.run(args)

def hyperparameter_optimization(args):
    hyperopt.run(args)


def parse_arguments(experiment):
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('dataset', type=str, help='Name of the dataset.')
    parser.add_argument('exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'Multitask', 'Joint', 'Probe',
                                 'TTI', 'Robustness', 'HyperparameterSearch'],
                        help='Name of experiment to run.')
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')

    #kd parameter
    parser.add_argument('--temperature', default=3, type=int,
                        help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float,
                        help='weight of feature loss')

    if experiment == 'Probe':
        return (probe.parse_arguments(parser),)

    elif experiment == 'TTI':
        return (tti.parse_arguments(parser),)

    elif experiment == 'Robustness':
        return (gen_cub_synthetic.parse_arguments(parser),)

    elif experiment == 'HyperparameterSearch':
        return (hyperopt.parse_arguments(parser),)

    else:
        parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
        parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
        parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
        parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
        parser.add_argument('-lr', type=float, help="learning rate")
        parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
        parser.add_argument('-pretrained', '-p', action='store_true',
                            help='whether to load pretrained model & just fine-tune')
        parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
        parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
        parser.add_argument('-use_attr', action='store_true',
                            help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
        parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
        parser.add_argument('-no_img', action='store_true',
                            help='if included, only use attributes (and not raw imgs) for class prediction')
        parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
        parser.add_argument('-weighted_loss', default='', # note: may need to reduce lr
                            help='Whether to use weighted loss for single attribute or multiple ones')
        parser.add_argument('-uncertain_labels', action='store_true',
                            help='whether to use (normalized) attribute certainties as labels')
        parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                            help='whether to apply bottlenecks to only a few attributes')
        parser.add_argument('-expand_dim', type=int, default=0,
                            help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
        parser.add_argument('-n_class_attr', type=int, default=2,
                            help='whether attr prediction is a binary or triary classification')
        parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
        parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
        parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
        parser.add_argument('-end2end', action='store_true',
                            help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
        parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
        parser.add_argument('-ckpt', default='', help='For retraining on both train + val set')
        parser.add_argument('-scheduler_step', type=int, default=1000,
                            help='Number of steps before decaying current learning rate by half')
        parser.add_argument('-normalize_loss', action='store_true',
                            help='Whether to normalize loss by taking attr_loss_weight into account')
        parser.add_argument('-use_relu', action='store_true',
                            help='Whether to include relu activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-use_sigmoid', action='store_true',
                            help='Whether to include sigmoid activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-connect_CY', action='store_true',
                            help='Whether to use concepts as auxiliary features (in multitasking) to predict Y')



        args = parser.parse_args()
        args.three_class = (args.n_class_attr == 3)
        return (args,)
