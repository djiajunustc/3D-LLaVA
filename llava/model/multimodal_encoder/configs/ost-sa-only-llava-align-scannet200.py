
num_semantic_classes = 200
num_instance_classes = 198
segment_ignore_index = [-1, 0, 2]
instance_ignore_index= -1

num_channels = 32

# model settings
model = dict(
    type="OmniSuperPointTransformer",
    query_thr=0.5,
    eval_mode="instance",
    num_classes=num_semantic_classes,
    stuff_classes=segment_ignore_index,
    backbone=dict(
        type="SpConvUNet",
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='ScanNetQueryDecoder_SA_Only_Dist_Bias',
        num_layers=3,
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=False,
        fix_attention=True,
        objectness_flag=False),
    )


visual_sampler = dict(
    type='VisualSampler_SA_Only_Dist_Bias',
    num_layers=3,
    num_instance_queries=0,
    num_semantic_queries=0,
    num_instance_classes=num_instance_classes,
    num_semantic_classes=num_semantic_classes,
    num_semantic_linears=1,
    in_channels=num_channels,
    d_model=256,
    num_heads=8,
    hidden_dim=1024,
    dropout=0.0,
    activation_fn='gelu',
    iter_pred=True,
    attn_mask=False,
    fix_attention=True,
    objectness_flag=False
    )

mask_decoder=dict(
    type='MaskDecoder_SA_Only',
    num_layers=3,
    num_instance_queries=0,
    num_semantic_queries=0,
    num_instance_classes=num_instance_classes,
    num_semantic_classes=num_semantic_classes,
    num_semantic_linears=1,
    in_channels=num_channels,
    d_model=256,
    num_heads=8,
    hidden_dim=1024,
    dropout=0.0,
    activation_fn='gelu',
    iter_pred=True,
    attn_mask=False,
    fix_attention=True,
    objectness_flag=False)

criteria=dict(
    type='PerceptionCriterion',
    ref_criterion=dict(
        type='ReferSegCriterion',
        matcher=dict(
            type='SparseReferMatcher',
            costs=[
                dict(type='MaskBCECost', weight=2.0),
                dict(type='MaskDiceCost', weight=2.0)
            ],
            topk=1),
        loss_weight=[0.0, 1.0/10, 1.0/10, 0.5/10],
        pos_weight=2,
        fix_dice_loss_weight=True,
        iter_matcher=True,
        fix_mean_loss=True))