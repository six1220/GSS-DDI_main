import os
# 设置PyTorch CUDA内存分配器配置，避免内存碎片化导致的OOM错误
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data

from dataset_pac.collator import collator
from models.model import MultiLevelDDI

from dataset_pac.dataset import Dataset
from utils.logging_utils import LOG, LOSS_FUNCTIONS
import utils.logging_utils as lu
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from torch.autograd import Variable
from utils import parse_utils
from utils.checkpoint_utils import get_checkpoint_filename, save_checkpoint, load_checkpoint

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

args = parse_utils.get_args().parse_args()
# Post-process gene expression arguments: convert string to proper types
if hasattr(args, 'use_cell_specific_expr') and isinstance(args.use_cell_specific_expr, str):
    args.use_cell_specific_expr = args.use_cell_specific_expr.lower() in ('true', '1', 'yes')
if hasattr(args, 'preferred_cell_line') and args.preferred_cell_line and args.preferred_cell_line.lower() == 'none':
    args.preferred_cell_line = None
# Update logging_utils dataset_path to match command line argument
lu.dataset_path = args.dataset_path
print(args)
torch.manual_seed(2)
np.random.seed(3)
# todo 五折没做，如果需要可以直接新建一个五折的数据集然后改dataset_name


def test(data_set, model):#接收数据加载器和模型
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for _, (d_node, d_in_degree, d_out_degree, p_node, p_in_degree, p_out_degree,
            label, d1, d2, mask_1, mask_2, d1_positions, d1_z, d1_batch, d2_positions, d2_z, d2_batch,
            (adj_1, nd_1, ed_1), (adj_2, nd_2, ed_2), d1_expr, d2_expr, d1_morgan, d2_morgan, tanimoto_similarity, 
            mask_1_gnn, mask_2_gnn) in enumerate(tqdm(data_set)):
        # 构建模型输入
        model_inputs = (
            d_node.cuda(), d_in_degree.cuda(), d_out_degree.cuda(),
            p_node.cuda(), p_in_degree.cuda(), p_out_degree.cuda(),
            d1.cuda(), d2.cuda(), mask_1.cuda(), mask_2.cuda(),
            d1_positions.cuda(), d1_z.cuda(), d1_batch.cuda(),
            d2_positions.cuda(), d2_z.cuda(), d2_batch.cuda(),
            adj_1.cuda(), nd_1.cuda(), ed_1.cuda(),
            adj_2.cuda(), nd_2.cuda(), ed_2.cuda()
        )
        
        # 添加基因表达特征（如果启用）
        if hasattr(args, 'use_gene_expr') and args.use_gene_expr:
            model_inputs = model_inputs + (d1_expr.cuda(), d2_expr.cuda())
        else:
            model_inputs = model_inputs + (None, None)
        
        # 添加 Morgan 指纹特征
        model_inputs = model_inputs + (d1_morgan.cuda(), d2_morgan.cuda())
        
        # 添加 Tanimoto 相似性特征
        model_inputs = model_inputs + (tanimoto_similarity.cuda(),)
        
        # 添加 GNN 掩码特征
        model_inputs = model_inputs + (torch.tensor(mask_1_gnn, dtype=torch.float32).cuda(), torch.tensor(mask_2_gnn, dtype=torch.float32).cuda())
        
        score, explanations = model(*model_inputs)

        label = Variable(torch.from_numpy(np.array(label - 1)).long()).cuda()#改改1
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(score, label)
        loss_accumulate += loss
        count += 1

        outputs = score.argmax(dim=1).detach().cpu().numpy() + 1
        label_ids = label.to('cpu').numpy() + 1

        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + outputs.flatten().tolist()

    loss = loss_accumulate / count

    accuracy = accuracy_score(y_label, y_pred)
    micro_precision = precision_score(y_label, y_pred, average='micro')
    micro_recall = recall_score(y_label, y_pred, average='micro')
    micro_f1 = f1_score(y_label, y_pred, average='micro')

    macro_precision = precision_score(y_label, y_pred, average='macro')
    macro_recall = recall_score(y_label, y_pred, average='macro')
    macro_f1 = f1_score(y_label, y_pred, average='macro')
    return accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss.item()





def main():

    loss_history = []
    training_logs = []  # 存储完整的训练日志
    start_epoch = 0
    # 检查是否有检查点文件
    checkpoint_dir = os.path.join(args.savemodels_dir, 'checkpoints')
    # 保留旧版本的检查点路径用于兼容性
    latest_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    
    # 如果设置了忽略检查点，清空训练历史
    if getattr(args, 'ignore_checkpoint', False):
        print('\n' + '='*80)
        print('⚠ 忽略检查点模式：将从头开始训练（清空训练历史）')
        print('='*80)
        # 可以选择删除检查点文件，或者只是不加载它们
        if os.path.exists(latest_checkpoint):
            print(f'发现检查点文件: {latest_checkpoint}')
            print('提示: 检查点文件不会被自动删除，但训练将从 epoch 0 开始')
        print('='*80 + '\n')
        # 强制从头开始训练
        args.model_path = None
        latest_checkpoint = None
    
    # 生成基于数据集和增强类型的检查点文件名
    aug_type = getattr(args, 'aug_type', None)
    checkpoint_filename = get_checkpoint_filename(args.dataset_path, args.dataset_name, aug_type, 'latest')
    dataset_specific_checkpoint = os.path.join(checkpoint_dir, checkpoint_filename)
    


    # 优先使用检查点恢复，其次使用 model_path
    # 如果设置了忽略检查点，直接从头开始训练
    if getattr(args, 'ignore_checkpoint', False):
        # 从头开始训练
        model = MultiLevelDDI(args)
        model = model.to(args.device)
        print('⚠ 忽略检查点：从头开始训练')
    elif args.model_path and os.path.exists(args.model_path):
        # 如果 model_path 是检查点文件
        if args.model_path.endswith('.pth') and 'checkpoint' in args.model_path:
            model = MultiLevelDDI(args)
            model = model.to(args.device)
            start_epoch, loss_history, training_logs = load_checkpoint(
                args.model_path, model, device=args.device, total_epochs=args.epochs,
                resume_from_prev_epoch=getattr(args, 'resume_from_prev_epoch', False),
                expected_dataset_path=args.dataset_path,
                expected_dataset_name=args.dataset_name,
                expected_aug_type=aug_type
            )
        else:
            # 旧的方式：直接加载模型
            model = torch.load(args.model_path, )
            model = model.to(args.device)
            print(f'从 {args.model_path} 加载模型（仅模型权重，不包括训练状态）')
            loss_history = []
            training_logs = []
    elif os.path.exists(dataset_specific_checkpoint):
        # 自动从数据集特定的检查点恢复
        model = MultiLevelDDI(args)
        model = model.to(args.device)
        start_epoch, loss_history, training_logs = load_checkpoint(
                dataset_specific_checkpoint, model, device=args.device, total_epochs=args.epochs,
                resume_from_prev_epoch=getattr(args, 'resume_from_prev_epoch', False),
                expected_dataset_path=args.dataset_path,
                expected_dataset_name=args.dataset_name,
                expected_aug_type=aug_type
            )
    elif latest_checkpoint is not None and os.path.exists(latest_checkpoint):
        # 兼容旧版本的检查点（没有数据集标识的）
        print('⚠ 发现旧版本的检查点（无数据集标识），尝试加载...')
        model = MultiLevelDDI(args)
        model = model.to(args.device)
        try:
            start_epoch, loss_history, training_logs = load_checkpoint(
                latest_checkpoint, model, device=args.device, total_epochs=args.epochs,
                resume_from_prev_epoch=getattr(args, 'resume_from_prev_epoch', False),
                expected_dataset_path=args.dataset_path,
                expected_dataset_name=args.dataset_name,
                expected_aug_type=aug_type
            )
        except ValueError as e:
            # 如果数据集不匹配，从头开始训练
            print(f'⚠ 无法使用旧检查点: {e}')
            print('从头开始训练...')
            start_epoch = 0
            loss_history = []
            training_logs = []
    else:
        # 从头开始训练
        model = MultiLevelDDI(args)
        model = model.to(args.device)
        start_epoch = 0
        loss_history = []
        training_logs = []

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, dim=0) 这句话真害人

    # 调整 num_workers 参数，使用多线程加载数据
    # 当 num_workers 为 0 时，设置为 4 以提高性能
    num_workers = args.num_workers if args.num_workers > 0 else 4
    
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': num_workers,
              'drop_last': True,
              'collate_fn': collator}

    # 构建数据文件路径
    if args.dataset_name:
        train_file = os.path.join(args.dataset_path, args.dataset_name, 'train.csv')
        val_file = os.path.join(args.dataset_path, args.dataset_name, 'val.csv')
        test_file = os.path.join(args.dataset_path, args.dataset_name, 'test.csv')
    else:
        # 如果 dataset_name 为空，直接使用 dataset_path 目录
        train_file = os.path.join(args.dataset_path, 'train.csv')
        val_file = os.path.join(args.dataset_path, 'val.csv')
        test_file = os.path.join(args.dataset_path, 'test.csv')
    
    # 读取数据文件
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)

    training_set = Dataset(train_data, 'train', args)
    validation_set = Dataset(val_data, 'val', args)
    testing_set = Dataset(test_data, 'test', args)

    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    testing_generator = data.DataLoader(testing_set, **params)  # 直接用params中的collate_fn替换dataloader的collate_fn 类似于重写

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 如果从检查点恢复，需要重新加载优化器状态
    if start_epoch > 0:
        if args.model_path and os.path.exists(args.model_path) and 'checkpoint' in args.model_path:
            checkpoint_path = args.model_path
        elif latest_checkpoint is not None and os.path.exists(latest_checkpoint):
            checkpoint_path = latest_checkpoint
        else:
            checkpoint_path = None
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
            if 'optimizer_state_dict' in checkpoint:
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                print('优化器状态已恢复')
    
    criterion = LOSS_FUNCTIONS[args.loss]
    # scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'], eta_min=args.min_lr)

    print('--- Go for Training ---')
    if start_epoch > 0:
        print(f'从 epoch {start_epoch + 1} 继续训练（共 {args.epochs} 个 epoch）')
    torch.backends.cudnn.benchmark = True
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    
    print(f'训练数据加载完成，开始训练...')
    print(f'训练集大小: {len(training_set)}')
    print(f'验证集大小: {len(validation_set)}')
    print(f'测试集大小: {len(testing_set)}')
    print(f'批次大小: {args.batch_size}')
    print(f'每个 epoch 的迭代次数: {len(training_generator)}')
    
    for epo in range(start_epoch, args.epochs):
        print(f'\n开始 Epoch {epo + 1}/{args.epochs}')
        model.train()
        start_time = time.time()
        epoch_start = time.time()
        
        for i, batch in enumerate(training_generator):
            try:
                # 检查批次是否为空
                if batch is None:
                    continue
                
                drug1_node, drug1_in_degree, drug1_out_degree, drug2_node, drug2_in_degree, drug2_out_degree, \
                target_tensor, d1_emb_tensor, d2_emb_tensor, mask_1_tensor, mask_2_tensor, d1_batch_positions, d1_batch_z, d1_batch, d2_batch_positions, d2_batch_z, d2_batch, \
                (adjacency_tensor_1, node_tensor_1, edge_tensor_1), (adjacency_tensor_2, node_tensor_2, edge_tensor_2), d1_expr_tensor, d2_expr_tensor, d1_morgan_tensor, d2_morgan_tensor, tanimoto_tensor, \
                d1_mask_gnn_tensor, d2_mask_gnn_tensor = batch
                
                opt.zero_grad()
                
                # 构建模型输入
                model_inputs = (
                    drug1_node.to(args.device), drug1_in_degree.to(args.device), drug1_out_degree.to(args.device),
                    drug2_node.to(args.device), drug2_in_degree.to(args.device), drug2_out_degree.to(args.device),
                    d1_emb_tensor.to(args.device), d2_emb_tensor.to(args.device), mask_1_tensor.to(args.device), mask_2_tensor.to(args.device),
                    d1_batch_positions.to(args.device), d1_batch_z.to(args.device), d1_batch.to(args.device),
                    d2_batch_positions.to(args.device), d2_batch_z.to(args.device), d2_batch.to(args.device),
                    adjacency_tensor_1.to(args.device), node_tensor_1.to(args.device), edge_tensor_1.to(args.device),
                    adjacency_tensor_2.to(args.device), node_tensor_2.to(args.device), edge_tensor_2.to(args.device)
                )
                
                # 添加基因表达特征（如果启用）
                if hasattr(args, 'use_gene_expr') and args.use_gene_expr:
                    model_inputs = model_inputs + (d1_expr_tensor.to(args.device), d2_expr_tensor.to(args.device))
                else:
                    model_inputs = model_inputs + (None, None)
                
                # 添加 Morgan 指纹特征
                model_inputs = model_inputs + (d1_morgan_tensor.to(args.device), d2_morgan_tensor.to(args.device))
                
                # 添加 Tanimoto 相似性特征
                model_inputs = model_inputs + (tanimoto_tensor.to(args.device),)
                
                # 添加 GNN 掩码特征
                model_inputs = model_inputs + (d1_mask_gnn_tensor.to(args.device), d2_mask_gnn_tensor.to(args.device))
                
                score, explanations = model(*model_inputs)  # torch tensor#传递给模型
                
                label = target_tensor.long().to(args.device)  # torch tensor
                # loss_fct = torch.nn.CrossEntropyLoss().cuda()
                loss = criterion(score, label)

                if torch.isnan(loss).any(): # 1. 检测NaN损失(not a number)
                    for param_group in opt.param_groups: # 2. 降低学习率
                        param_group['lr'] = param_group['lr'] / 10
                    # 3. 重新加载模型（回退到之前保存的版本）
                    model = torch.load(args.savemodels_dir + type(model).__name__, weights_only=False)
                    # 4. 打印信息并跳出当前epoch
                    print('In Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' lr decrease')
                    break
                # print(label.shape,label)
                # print(loss)
                # assert False
                # loss = loss_fct(score, label)
                loss_history.append(loss.item())# 记录损失值

                loss.backward()# 反向传播
                
                torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
                # 用于裁剪梯度，梯度裁剪是一种正则化技术，用于防止在训练深度学习模型时发生梯度爆炸
                opt.step()# 更新参数
                
                # 清理中间变量和缓存
                del score, loss, model_inputs, explanations
                torch.cuda.empty_cache()
                
            except RuntimeError as e:#OOM处理
                if 'out of memory' in str(e).lower():
                    print('\n' + '='*80)
                    print(f'[严重错误] CUDA 内存不足 (Out of Memory)')
                    print('='*80)
                    print(f'训练位置: Epoch {epo + 1}/{args.epochs}, Iteration {i}')
                    print(f'错误详情: {str(e)}')
                    print('-'*80)
                    
                    # 清理缓存并打印 GPU 内存使用情况
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(args.device) / 1024**3
                        reserved = torch.cuda.memory_reserved(args.device) / 1024**3
                        total = torch.cuda.get_device_properties(args.device).total_memory / 1024**3
                        print(f'GPU 内存使用情况:')
                        print(f'  - 已分配: {allocated:.2f} GB')
                        print(f'  - 已保留: {reserved:.2f} GB')
                        print(f'  - 总容量: {total:.2f} GB')
                        print(f'  - 可用: {total - reserved:.2f} GB')
                    
                    # 尝试保存检查点
                    print('-'*80)
                    print('正在尝试保存检查点...')
                    try:
                        if args.savemodel:
                            save_checkpoint(model, opt, epo + 1, loss_history, checkpoint_dir,
                                          dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                          aug_type=aug_type, is_best=False)
                            print('✓ 检查点已保存，可以从断点继续训练')
                        else:
                            print('⚠ 检查点保存已禁用 (savemodel=False)')
                    except Exception as save_error:
                        print(f'✗ 保存检查点失败: {save_error}')
                    
                    print('='*80)
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print('\n训练已停止。请解决内存问题后重新运行。\n')
                    
                    # 停止训练
                    raise SystemExit(1)
                else:
                    # 其他 RuntimeError
                    print('\n' + '='*80)
                    print(f'[错误] 训练过程中发生运行时错误')
                    print('='*80)
                    print(f'训练位置: Epoch {epo + 1}/{args.epochs}, Iteration {i}')
                    print(f'错误类型: {type(e).__name__}')
                    print(f'错误详情: {str(e)}')
                    print('-'*80)
                    
                    # 尝试保存检查点
                    print('正在尝试保存检查点...')
                    try:
                        if args.savemodel:
                            save_checkpoint(model, opt, epo + 1, loss_history, checkpoint_dir,
                                          dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                          aug_type=aug_type, is_best=False)
                            print('✓ 检查点已保存，可以从断点继续训练')
                        else:
                            print('⚠ 检查点保存已禁用 (savemodel=False)')
                    except Exception as save_error:
                        print(f'✗ 保存检查点失败: {save_error}')
                    
                    print('='*80)
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print('\n训练已停止。请解决运行时错误后重新运行。\n')
                    
                    # 停止训练
                    raise SystemExit(1)
            except IndexError as e:
                # 捕获索引越界错误
                print('\n' + '='*80)
                print(f'[错误] 训练过程中发生索引越界错误')
                print('='*80)
                print(f'训练位置: Epoch {epo + 1}/{args.epochs}, Iteration {i}')
                print(f'错误类型: {type(e).__name__}')
                print(f'错误详情: {str(e)}')
                print('-'*80)
                
                # 输出调试信息
                print('调试信息:')
                print(f'批次大小: {args.batch_size}')
                print(f'当前迭代: {i}')
                try:
                    if 'drug1_node' in locals():
                        print(f'drug1_node形状: {drug1_node.shape}')
                    if 'drug2_node' in locals():
                        print(f'drug2_node形状: {drug2_node.shape}')
                    if 'd1_emb_tensor' in locals():
                        print(f'd1_emb_tensor形状: {d1_emb_tensor.shape}')
                    if 'd2_emb_tensor' in locals():
                        print(f'd2_emb_tensor形状: {d2_emb_tensor.shape}')
                    if 'd1_morgan_tensor' in locals():
                        print(f'd1_morgan_tensor形状: {d1_morgan_tensor.shape}')
                    if 'd2_morgan_tensor' in locals():
                        print(f'd2_morgan_tensor形状: {d2_morgan_tensor.shape}')
                    if 'tanimoto_tensor' in locals():
                        print(f'tanimoto_tensor形状: {tanimoto_tensor.shape}')
                except Exception as debug_error:
                    print(f'调试信息获取失败: {debug_error}')
                
                # 尝试保存检查点
                print('-'*80)
                print('正在尝试保存检查点...')
                try:
                    if args.savemodel:
                        save_checkpoint(model, opt, epo + 1, loss_history, checkpoint_dir,
                                      dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                      aug_type=aug_type, is_best=False)
                        print('✓ 检查点已保存，可以从断点继续训练')
                    else:
                        print('⚠ 检查点保存已禁用 (savemodel=False)')
                except Exception as save_error:
                    print(f'✗ 保存检查点失败: {save_error}')
                
                print('='*80)
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print('\n训练已停止。请解决索引越界错误后重新运行。\n')
                
                # 停止训练
                raise SystemExit(1)
            except Exception as e:
                # 捕获其他所有错误
                print('\n' + '='*80)
                print(f'[错误] 训练过程中发生异常')
                print('='*80)
                print(f'训练位置: Epoch {epo + 1}/{args.epochs}, Iteration {i}')
                print(f'错误类型: {type(e).__name__}')
                print(f'错误详情: {str(e)}')
                print('-'*80)
                
                # 打印堆栈信息
                import traceback
                print('堆栈信息:')
                print(traceback.format_exc())
                
                # 尝试保存检查点
                print('-'*80)
                print('正在尝试保存检查点...')
                try:
                    if args.savemodel:
                        save_checkpoint(model, opt, epo + 1, loss_history, checkpoint_dir,
                                      dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                      aug_type=aug_type, is_best=False)
                        print('✓ 检查点已保存，可以从断点继续训练')
                    else:
                        print('⚠ 检查点保存已禁用 (savemodel=False)')
                except Exception as save_error:
                    print(f'✗ 保存检查点失败: {save_error}')
                
                print('='*80)
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print('\n训练已停止。请解决错误后重新运行。\n')
                
                # 停止训练
                raise SystemExit(1)
                
            # scheduler.step()
            end_time = time.time()
            
            # 修改日志输出条件，确保迭代0时也能输出
            if i % 100 == 0:
                if loss_history:
                    log_msg = 'Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                        loss_history[-1]) + ' use time ' + str(end_time - start_time) + 's'
                    print(log_msg, flush=True)
                    training_logs.append(log_msg)  # 保存训练日志
            start_time = end_time
        
        epoch_end = time.time()
        print(f'Epoch {epo + 1} 完成，总用时: {epoch_end - epoch_start:.2f}s')

        with torch.set_grad_enabled(False):#禁用梯度计算，减少内存占用和计算时间
            model.eval()
            # 在评估前记录模型文件的修改时间，用于判断是否是最佳模型
            is_best = False
            if args.savemodel and hasattr(args, 'savemodels_dir') and os.path.exists(args.savemodels_dir):
                model_file = os.path.join(args.savemodels_dir, type(model).__name__)
                file_time_before = os.path.getmtime(model_file) if os.path.exists(model_file) else 0
            else:
                file_time_before = 0
            
            # 捕获评估输出的日志
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            LOG[args.logging](model, training_generator, validation_generator, testing_generator, criterion, epo, args)
            
            # 获取评估输出并恢复stdout
            eval_output = captured_output.getvalue()
            sys.stdout = old_stdout
            print(eval_output, end='')  # 正常输出评估结果（不添加额外换行）
            # 保存评估日志（按行分割，去除空行）
            eval_lines = [line for line in eval_output.rstrip().split('\n') if line.strip()]
            training_logs.extend(eval_lines)
            
            # 检查是否更新了最佳模型（只考虑测试集性能）
            is_best = False
            try:
                import utils.logging_utils as lu
                if hasattr(lu, 'g'):
                    # 只检查 best testing epoch 是否等于当前 epoch
                    if hasattr(lu.g, 'best_testing_epoch') and lu.g.best_testing_epoch == epo + 1:
                        is_best = True
                        print(f'[最佳模型] Epoch {epo + 1} 达到了最佳测试性能，模型已保存')
            except Exception as e:
                print(f'检查最佳模型时出错: {e}')
            
            # accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss = test(validation_generator, model)
            # print("[Validation metrics]: loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
            #     loss, accuracy, macro_precision, macro_recall, macro_f1))
            # if accuracy > max_auc:
            #    # torch.save(model, 'save_model/' + str(accuracy) + '_model.pth')
            #     torch.save(model, 'save_model/best_model.pth')
            #     model_max = copy.deepcopy(model)
            #     max_auc = accuracy
            #     print("*" * 30 + " save best model " + "*" * 30)

        # 每个 epoch 结束后保存检查点
        if args.savemodel:
            save_checkpoint(model, opt, epo + 1, loss_history, checkpoint_dir, 
                          dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                          aug_type=aug_type, is_best=is_best, training_logs=training_logs)
        
        # torch.cuda.empty_cache()

    # print('\n--- Go for Testing ---')
    # try:
    #     with torch.set_grad_enabled(False):
    #         accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss  = test(testing_generator, model_max)
    #         print("[Testing metrics]: loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
    #             loss, accuracy, macro_precision, macro_recall, macro_f1))
    # except:
    #     print('testing failed')
    return model, loss_history


'''
nohup python -u train_binary.py > new3_adataset_lr-5_1.log 2>&1 &
new1: AMDE + Molormer
new2: AMDE + Molormer + decoder-->BilinearDecoder  ×
new3: AMDE + Molormer + concat-->乘以权重
'''

if __name__ == '__main__':
    main()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))