import torch       # ���C�u�����uPyTorch�v��torch�p�b�P�[�W���C���|�[�g
import torch.nn as nn  # �u�j���[�����l�b�g���[�N�v���W���[���̕ʖ���`


# �utorch.nn.Module�N���X�̃T�u�N���X���v�ɂ�郂�f���̒�`
class FeatToParam(nn.Module):
    def __init__(self):
        super(FeatToParam, self).__init__()
        # �w�ilayer�F���C���[�j���`
        self.coord_head2 = nn.Conv2d(32, 34, 3)

        #self.layer1 = nn.Linear(  # Linear�́u�S�����w�v���w��
        #    INPUT_FEATURES,       # �f�[�^�i�����j�̓��̓��j�b�g��
        #    OUTPUT_NEURONS)       # �o�͌��ʂւ̏o�̓��j�b�g��

    def forward(self, input):
        # �t�H���[�h�p�X���`
        output = self.coord_head2(input)
        output = output.reshape(-1, 34) 

        #output = activation(self.layer1(input))  # �������֐��͕ϐ��Ƃ��Ē�`
        # �u�o�́��������֐��i��n�w�i���́j�j�v�̌`���ŋL�q����B
        # �w�ilayer�j���d�˂�ꍇ�́A���l�̋L�q�𑱂���΂悢�i��3��j�B
        # �u�o�́ioutput�j�v�͎��̑w�ilayer�j�ւ́u���́iinput�j�v�Ɏg���B
        # ����ł͓��͂��o�͂��ux�v�Ɠ����ϐ����ŋL�q����i����Ĉȉ��ł́ux�v�Ə����j
        return output

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
