_base_ = [
    '../_base_/models/energy_potential.py',
    '../_base_/datasets/metacloth_dynamic.py',
    '../_base_/schedules/adam_step.py',
    '../_base_/default_runtime.py'
]

# New customized option
step = 2+1 ## Extra one for previous input for dissipate state

# Custom dataset
data = dict(
    train=dict(
        env_cfg=dict(
            pad_f_connect=True,
            edgewise=True,
            val_seq=800,
            noise_std=0.00001,
            noise_step=10, # split the noise into 10 sub interval
            omit_frame=step+1, # Since it's auto regressive and use future frame vel
            step=step,),),
    val=dict(
        env_cfg=dict(
            val_seq=200,
            pad_f_connect=True,
            step=step,),),
    test=dict(
        env_cfg=dict(
            val_seq=200,
            pad_f_connect=True,
            rollout=False,
            step=step),),)

## learning policy
runner = dict(type='EpochRunner', max_epochs=6)
