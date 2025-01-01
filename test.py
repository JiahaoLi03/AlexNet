import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet


def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                             download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)

    return test_dataloader


def test_model_process(model, test_dataloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    test_correct = 0.0

    test_num = 0

    # 类别 FashionMNIST 数据中每个类别的标签
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 只进行前向传播，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            output = model(test_data_x)

            # 预测类别的下标
            pre_lab = torch.argmax(output, dim=1)

            # 输出预测结果
            # item() 方法将一个单元素的 Tensor 转换为 Python 标量，使得我们可以直接使用这个值
            pre_val = pre_lab.item()
            # 真实标签
            ground_truth = test_data_y.item()

            print("预测值：", pre_val, "------", "真实值：", ground_truth)
            print("预测值：", classes[pre_val], "------", "真实值：", classes[ground_truth])

            test_correct += torch.sum(pre_lab == test_data_y.data)

            test_num += test_data_x.size(0)

    test_acc = test_correct.double().item() / test_num
    print("测试的准确率为：", test_acc)


if __name__  == '__main__':

    AlexNet = AlexNet()

    AlexNet.load_state_dict(torch.load('best_model.pth'))

    test_dataloader = test_data_process()

    test_model_process(AlexNet, test_dataloader)

