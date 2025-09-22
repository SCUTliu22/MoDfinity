class YourModel(nn.Module):
    # 创建你的模型,这里以声音和图像模态举例，你需要添加特征编码器YourImgModel和YourAudioModel，Class_Transformer并将特征编码器的forward函数改写。Class_Transformer可以是全连接层，也可以是其他将特征转化为class的网络
    #     def forward(self, input):
    #       feature = your_backbone(input)
    #       output = your_fc(feature)
    #       return output, feature
    def __init__(self, in_planes1=1, in_planes2=3, num_cls=28):
        super(YourModel, self).__init__()
        self.img_model = YourImgModel(in_planes2, num_cls)
        self.audio_model = YourAudioModel(in_planes1, num_cls)
        self.cls_transformer = Class_Transformer(YourFeatureDim, num_cls)
        self.num_cls = num_cls
        self.arrange_label = torch.tensor(range(num_cls)).to(device)

    def forward(self, img_in, audio_in):
        img_output, img_feature = self.img_model(img_in)
        audio_output, audio_feature = self.audio_model(audio_in)
        return img_output, audio_output, img_feature, audio_feature

    def get_output(self, img_in, audio_in):
        # 修正方法调用，使用forward方法并正确处理返回值
        img_output, _ = self.img_model(img_in)
        audio_output, _ = self.audio_model(audio_in)
        return img_output, audio_output

    def get_one_hot(self, targets):
        target_one_hot = F.one_hot(targets, num_classes=self.num_cls)
        target_one_hot = target_one_hot.to(torch.float32)
        return target_one_hot

    def get_cls_feature(self, target):
        target_one_hot = self.get_one_hot(target)
        label_feature = self.cls_transformer(target_one_hot)
        return label_feature

    def get_similarity(self, img_feature, audio_feature):
        #   mo的输出，以亲和度度量为标准，判断输出结果
        l_feature = self.get_cls_feature(self.arrange_label)
        img_feature /= img_feature.norm(dim=-1, keepdim=True)
        audio_feature /= audio_feature.norm(dim=-1, keepdim=True)
        l_feature /= l_feature.norm(dim=-1, keepdim=True)
        img_similarity = (img_feature @ l_feature.T).softmax(dim=-1)
        audio_similarity = (audio_feature @ l_feature.T).softmax(dim=-1)
        return img_similarity, audio_similarity

class LossManager:
    # 构建类时提前告知batch_size和cuda
    # 在pretrain和retrain阶段调用compute_pretrain_losses和compute_retrain_losses，需要传入模型，标签（也可以是伪标签），各个模态的输入数据
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.ones_m = torch.ones((batch_size, batch_size)).to(device)
        self.eye_m = torch.eye(batch_size).to(device)
        self.loss_ce = nn.CrossEntropyLoss()
        self.ls_lambda = 5.0  # moml默认值，可以通过参数修改,样本越少值越小
        self.beta = 6.0  # 蒸mod损失的权重，默认值为6,样本越少值越小
    
    def get_label_delabel_maxtri(self, label):
        label_repeat = label.clone().unsqueeze(0).repeat(self.batch_size, 1)
        label_repeat_T = label_repeat.T
        label_maxtri_ori = label_repeat - label_repeat_T
        label_maxtri = torch.where(label_maxtri_ori != 0, 0, 1)
        return label_maxtri
    
    def loss_mo(self, i_representations, a_representations, batch_size, label_maxtri):
        temperature = 0.05

        i_representations = F.normalize(i_representations, dim=1)  # (bs, dim)  --->  (bs, dim)
        a_representations = F.normalize(a_representations, dim=1)  # (bs, dim)  --->  (bs, dim)

        similarity_matrix = F.cosine_similarity(i_representations.unsqueeze(1), a_representations.unsqueeze(0),
                                            dim=2)  # simi_mat: (bs, bs)
        nominator = label_maxtri * torch.exp(similarity_matrix / temperature)
        denominator =  torch.exp(similarity_matrix / temperature)  # bs, bs
        if torch.sum(nominator) == 0 or torch.sum(denominator) == 0:
            return 0
        loss_partial = -torch.log(
            torch.sum(nominator, dim=1) / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (batch_size) * 2
        return loss
    
    def compute_retrain_losses(self, model, imgs, audios, targets):
        # 前向传播
        img_output, audio_output, img_feature, audio_feature = model(imgs, audios)
        
        # 获取实际批次大小
        actual_batch_size = targets.size(0)
      
        # 获取标签矩阵
        label_maxtri = self.get_label_delabel_maxtri()
        
        # 获取类别特征
        if isinstance(model, nn.DataParallel):
            label_feature = model.module.get_cls_feature(targets)
        else:
            label_feature = model.get_cls_feature(targets)
        
        # 计算分类损失
        ce_img_loss = self.loss_ce(img_output, targets)
        ce_audio_loss = self.loss_ce(audio_output, targets)
        
        # 计算moml损失 - 使用detach()防止梯度传递到特征提取器
        limg_moml = self.loss_mo(
            img_feature.detach(), label_feature, actual_batch_size, label_maxtri
        )
        laudio_moml = self.loss_mo(
            audio_feature.detach(), label_feature, actual_batch_size, label_maxtri
        )
        
        # 计算mod损失
        distill_img_mod = self.loss_mo(
            img_feature, audio_feature.detach(), actual_batch_size, label_maxtri
        )
        distill_audio_mod = self.loss_mo(
            img_feature.detach(), audio_feature, actual_batch_size, label_maxtri
        )
        
        total_loss = ce_img_loss + self.beta * distill_img_mod + \
                     ce_audio_loss + self.beta * distill_audio_mod + \
                     self.ls_lambda * (limg_moml + laudio_moml)
    
        return total_loss


    def compute_pretrain_losses(self, model, imgs, audios, targets):
        # 前向传播
        img_output, audio_output, img_feature, audio_feature = model(imgs, audios)
        
        # 获取实际批次大小
        actual_batch_size = targets.size(0)
      
        # 获取标签矩阵
        label_maxtri = self.get_label_delabel_maxtri()
        
        # 获取类别特征
        if isinstance(model, nn.DataParallel):
            label_feature = model.module.get_cls_feature(targets)
        else:
            label_feature = model.get_cls_feature(targets)
        
        # 计算分类损失
        ce_img_loss = self.loss_ce(img_output, targets)
        ce_audio_loss = self.loss_ce(audio_output, targets)
        
        # 计算moml损失 - 使用detach()防止梯度传递到特征提取器
        limg_moml = self.loss_mo(
            img_feature.detach(), label_feature, actual_batch_size, label_maxtri
        )
        laudio_moml = self.loss_mo(
            audio_feature.detach(), label_feature, actual_batch_size, label_maxtri
        )

        
        total_loss = ce_img_loss +   \
                     ce_audio_loss +  \
                     self.ls_lambda * (limg_moml + laudio_moml)
    
        return total_loss
