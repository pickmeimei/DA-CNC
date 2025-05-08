import time

from parse_args import *
from utils import *
from dgi import *
import torch.nn.functional as F
import loss_func

###和扩散表示进行对比（1.原图和扩散图相当于两个视图，从这两个视图中学习到的表示之间的一致性最大化子图表示（最大似然估计），缓解稀疏）
# 2.原图和跨域的扩散图进行对比，之前的模型只考虑到一致性，使用对抗学习的，
# 没有考虑到子图（从一个大的网络中提取中的两个网络）之间应该有互补性，所以有跨域对比）
##源和目标

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    adj_s, feature_s, label_s, idx_tot_s, diff_s = load_data_diff(dataset=args.source_dataset + '.mat', device=device,
                                                                  seed=args.seed, is_blog=args.is_blog)
    adj_t, feature_t, label_t, idx_tot_t, diff_t = load_data_diff(dataset=args.target_dataset + '.mat', device=device,
                                                                  seed=args.seed, is_blog=args.is_blog)
    n_samples = args.n_samples.split(',')
    output_dims = args.output_dims.split(',')
    emb_model = GraphSAGE(**{
        "aggregator_class": aggregator_lookup[args.aggregator_class],
        "input_dim": feature_s.shape[1],
        "layer_specs": [
            {
                "n_sample": int(n_samples[0]),
                "output_dim": int(output_dims[0]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[1]),
                "output_dim": int(output_dims[1]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[-1]),
                "output_dim": int(output_dims[-1]),
                "activation": F.relu,
            }
        ],
        "device": device
    }).to(device)
    cly_model = Cly_net(2 * int(output_dims[-1]), label_s.shape[1], args.arch_cly).to(device)
    disc_model = Disc(2 * int(output_dims[-1]) * label_s.shape[1], args.arch_disc, 1).to(device)

    # define the optimizers
    total_params = list(emb_model.parameters()) + list(cly_model.parameters()) + list(disc_model.parameters())
    dgi_model = DGI(2 * int(output_dims[-1])).to(device)
    mi_model = MI(2 * int(output_dims[-1])).to(device)
    total_params += list(dgi_model.parameters())
    total_params += list(mi_model.parameters())
    cly_optim = torch.optim.Adam(total_params, lr=args.lr_cly, weight_decay=args.weight_decay)
    lr_lambda = lambda epoch: (1 + 10 * float(epoch) / args.epochs) ** (-0.75)
    scheduler = torch.optim.lr_scheduler.LambdaLR(cly_optim, lr_lambda=lr_lambda)
    best_micro_f1, best_macro_f1, best_epoch = 0, 0, 0
    num_batch = round(max(feature_s.shape[0] / (args.batch_size / 2), feature_t.shape[0] / (args.batch_size / 2)))
    for epoch in range(args.epochs):
        if epoch >= 0:
            t0 = time.time()
        s_batches = batch_generator(idx_tot_s, int(args.batch_size / 2))
        t_batches = batch_generator(idx_tot_t, int(args.batch_size / 2))
        emb_model.train()
        cly_model.train()
        disc_model.train()
        dgi_model.train()
        mi_model.train()
        p = float(epoch) / args.epochs
        grl_lambda = min(2. / (1. + np.exp(-10. * p)) - 1, 0.2)
        for iter in range(num_batch):
            b_nodes_s = next(s_batches)
            b_nodes_t = next(t_batches)
            source_features, cly_loss_s = do_iter(emb_model, cly_model, adj_s, feature_s, label_s, idx=b_nodes_s,
                                                  is_social_net=args.is_social_net)

            diff_source_features, _ = do_iter(emb_model, cly_model, diff_s, feature_s, label_s, idx=b_nodes_s,
                                                  is_social_net=args.is_social_net)
            target_features, _ = do_iter(emb_model, cly_model, adj_t, feature_t, label_t, idx=b_nodes_t,
                                         is_social_net=args.is_social_net)

            diff_target_features, _ = do_iter(emb_model, cly_model, diff_t, feature_t, label_t, idx=b_nodes_t,
                                         is_social_net=args.is_social_net)
            shuf_idx_s = np.arange(label_s.shape[0])
            np.random.shuffle(shuf_idx_s)
            shuf_feat_s = feature_s[shuf_idx_s, :]
            shuf_idx_t = np.arange(label_t.shape[0])
            np.random.shuffle(shuf_idx_t)
            shuf_feat_t = feature_t[shuf_idx_t, :]

            diff_source_feats = emb_model(b_nodes_s, diff_s, feature_s)
            neg_source_feats = emb_model(b_nodes_s, adj_s, shuf_feat_s)
            neg_target_feats = emb_model(b_nodes_t, adj_t, shuf_feat_t)
            #计算互信息
            # pos_scores = mi_model(source_features, diff_source_features)  # 正样本的互信息得分
            # neg_scores = mi_model(source_features, neg_source_feats)# 负样本的互信息得分
            # pos_scores_t = mi_model(target_features, diff_target_features)  # 正样本的互信息得分
            # neg_scores_t = mi_model(target_features, neg_target_feats)  # 负样本的互信息得分
            # pos_scores = torch.clamp(torch.sigmoid(pos_scores), min=1e-7, max=1 - 1e-7)
            # neg_scores = torch.clamp(torch.sigmoid(neg_scores), min=1e-7, max=1 - 1e-7)
            # pos_scores_t = torch.clamp(torch.sigmoid(pos_scores_t), min=1e-7, max=1 - 1e-7)
            # neg_scores_t = torch.clamp(torch.sigmoid(neg_scores_t), min=1e-7, max=1 - 1e-7)

            #跨域对比
            logits_s_cross = dgi_model(neg_source_feats, source_features, diff_target_features)#es和Et（diff)正对，扰乱es负对
            logits_t_cross = dgi_model(neg_target_feats, target_features, diff_source_features)

            labels_dgi = torch.cat(
                [torch.zeros(int(args.batch_size / 2)), torch.ones(int(args.batch_size / 2))]).unsqueeze(0).to(device)

            #两个视图之间的互信息最大化
            #计算损失：最大化正样本的对数似然，最小化负样本的对数似然
            # mi_loss_s = args.mi_param * (
            #             -torch.mean(torch.log(torch.sigmoid(pos_scores)) + torch.mean(torch.log(1 - torch.sigmoid(neg_scores)))))
            # mi_loss_t = args.mi_param * (
            #     -torch.mean(
            #         torch.log(torch.sigmoid(pos_scores_t)) + torch.mean(torch.log(1 - torch.sigmoid(neg_scores_t)))))
            # mi_loss =mi_loss_s+mi_loss_t

            #gpt
            # mi_loss = args.mi_param * (
            #     -torch.mean(
            #         torch.log(torch.sigmoid(pos_scores))) + torch.mean(torch.log(1 - torch.sigmoid(neg_scores))))

            #跨域对比loss
            dgi_loss_cross = args.dgi_param * (
                    F.binary_cross_entropy_with_logits(logits_s_cross, labels_dgi) + F.binary_cross_entropy_with_logits(
                logits_t_cross, labels_dgi))

            features = torch.cat((source_features, target_features), 0)
            outputs = cly_model(features)
            softmax_output = nn.Softmax(dim=1)(outputs)
            domain_loss = args.cdan_param * loss_func.CDAN([features, softmax_output], disc_model, None, grl_lambda,
                                                           None, device=device)


            loss = cly_loss_s + domain_loss + dgi_loss_cross
            # loss = cly_loss_s + mi_loss + dgi_loss_cross
            # loss = cly_loss_s + domain_loss
            # loss = cly_loss_s + mi_loss + domain_loss + dgi_loss_cross
            cly_optim.zero_grad()
            loss.backward()
            cly_optim.step()

        emb_model.eval()
        cly_model.eval()
        cly_loss_bat_s, micro_f1_s, macro_f1_s, embs_whole_s, targets_whole_s = evaluate(emb_model, cly_model, adj_s,
                                                                                         feature_s, label_s,
                                                                                         idx_tot_s, args.batch_size,
                                                                                         mode='test',
                                                                                         is_social_net=args.is_social_net)
        print("epoch {:03d} | source loss {:.4f} | source micro-F1 {:.4f} | source macro-F1 {:.4f}".
              format(epoch, cly_loss_bat_s, micro_f1_s, macro_f1_s))
        cly_loss_bat_t, micro_f1_t, macro_f1_t, embs_whole_t, targets_whole_t = evaluate(emb_model, cly_model, adj_t,
                                                                                         feature_t, label_t,
                                                                                         idx_tot_t, args.batch_size,
                                                                                         mode='test',
                                                                                         is_social_net=args.is_social_net)
        print("target loss {:.4f} | target micro-F1 {:.4f} | target macro-F1 {:.4f}".format(cly_loss_bat_t, micro_f1_t,
                                                                                            macro_f1_t))
        if (micro_f1_t + macro_f1_t) > (best_micro_f1 + best_macro_f1):
            best_micro_f1 = micro_f1_t
            best_macro_f1 = macro_f1_t
            best_epoch = epoch
            print('saving model...')
        scheduler.step()


    print("Time(s) {:.4f} ".
          format(time.time() - t0))
    print("test metrics on target graph:")
    print('---------- random seed: {:03d} ----------'.format(args.seed))
    print("micro-F1 {:.4f} | macro-F1 {:.4f}".format(best_micro_f1, best_macro_f1))

    f.write('best epoch: %d \t best micro_f1: %.6f \t best macro_f1: %.6f \n' % (best_epoch, best_micro_f1, best_macro_f1))
    f.flush()


if __name__ == '__main__':
    args = parse_args()

    f = open('output/' + args.source_dataset + '_' + args.target_dataset + '.txt', 'a+')
    f.write('\n\n\n{}\n'.format(args))
    f.flush()

    device = torch.device('cuda:' + str(args.device))
    print(device)
    main(args)
