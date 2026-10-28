"""Task-token count changes must propagate through representation and policy."""
import numpy as np
import pytest
import torch

from models.task_context_teacher import TaskContextTeacher
from models.task_context_encoder import TaskContextEncoder, belief_loss
from models.task_context_observer import TaskContextObserver


@pytest.mark.parametrize('tokens',[1,2,4,8])
def test_teacher_student_and_policy_token_count(tokens,tmp_path,monkeypatch):
    from models.align_intention import ALIGNIntentionModel
    from tests.test_task_context_policy import FakeVision
    monkeypatch.setattr('models.align_intention.VisionEncoder',FakeVision)
    teacher=TaskContextTeacher(cameras=1,width=16,layers=1,heads=4,tokens=tokens,
        token_dim=8,pooling='quarters',decoder_style='temporal',diffusion_width=8)
    batch=dict(features=torch.randn(1,4,1,5,768),states=torch.randn(1,4,10),
        times=torch.arange(4)[None].float(),lengths=torch.tensor([4]),anchors=torch.tensor([[0,2]]))
    result=teacher(batch)
    assert result['tokens'].shape==(1,2,tokens,8)
    torch.testing.assert_close(result['tokens'][:,0],result['tokens'][:,1])
    student=TaskContextEncoder(width=16,tokens=tokens,token_dim=8,layers=1,
        execution_dim=8,visual_width=768,d_state=4,d_conv=3,expand=2,horizons=[1,2])
    summaries=torch.randn(1,4,16)
    predictions=student(summaries,batch['lengths'],batch['anchors'])
    assert predictions['means'].shape==(1,2,4,tokens,8)  # Four hypotheses, independent of slot count.
    loss=belief_loss(student,predictions,result['tokens'][:,0].detach())
    loss.backward();assert torch.isfinite(loss) and student.means.weight.grad.abs().sum()>0
    observer=TaskContextObserver(dict(cameras=1,raw_dim=768,width=16),student.config)
    observer.encoder.load_state_dict(student.state_dict())
    path=tmp_path/'observer.pt'
    torch.save(dict(schema='align-causal-task-observer-v1',summary_config=observer.summary_config,
        encoder_config=student.config,summary=observer.summary.state_dict(),encoder=observer.encoder.state_dict(),
        state_mean=observer.state_mean,state_std=observer.state_std,metadata=dict(fps=20.)),path)
    def build(requested):
        model=ALIGNIntentionModel(state_dim=4,compressed_dim=4,num_cameras=1,mamba_output_dim=0,
            head_d_model=8,chunk_size=4,task_context_checkpoint=str(path),task_context_tokens=requested)
        model._build_head_and_bank(1024)
        return model.eval()
    model=build(tokens)
    visual,state,_,cache,payload=model.encode_step(torch.randn(1,257,768),torch.randn(1,7),produce_intent=True)
    assert payload.shape==(1,tokens+1,8)
    assert model.intention_head.context_tokens==tokens
    assert model.intention_head.context_task[0].normalized_shape==(tokens*8,)
    condition=model.action_condition(visual[:,None],state[:,None],payload)
    assert torch.isfinite(condition).all()
    restored=build(None);restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.action_condition(visual[:,None],state[:,None],payload),condition)
    with pytest.raises(ValueError,match='pretrain a matching'):
        build(tokens+1)


def test_token_probe_views_follow_count():
    from tools.probe_task_context_tokens import feature_views
    arrays=dict(tokens=np.zeros((2,1,8,128)),execution=np.zeros((2,1,128)),
                times=np.zeros((2,1)),mode=np.zeros((2,1),dtype=int),
                summary=np.zeros((2,1,16)),states=np.zeros((2,1,10)),cls=np.zeros((2,1,768)))
    views=feature_views(arrays,0)
    assert views['task_bundle'].shape==(2,1024)
    assert all(f'task_token_{i}' in views for i in range(1,9))


def test_policy_count_argument_requires_matching_context():
    from scripts.run_intention_ablation import parse_args
    with pytest.raises(SystemExit):parse_args(['--output','/tmp/not-created','--num-task-tokens','0'])
    args=parse_args(['--output','/tmp/not-created','--num-task-tokens','8',
                     '--task-context-checkpoint','/tmp/observer.pt'])
    assert args.num_task_tokens==8
