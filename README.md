# Neural Style Transfer и P²-GAN: поређење два метода преноса стила

Идеја овог пројекта је да се имплементирају и упореде две методе преноса стила, Neural Style Transfer из рада [*A Neural Algorithm of Artistic Style*](https://arxiv.org/abs/1508.06576)
(Gatys et al., 2015) као један од кључних радова у области преноса стила, и P²-GAN из рада [*P²-GAN: Efficient Style Transfer Using Single Style Image*](https://arxiv.org/abs/2001.07466) (Zheng & Liu, 2020) 
као један занимљив приступ примени генеративних супарничких мрежа на проблем преноса стила.

![repo](https://github.com/qkolj/style-transfer/blob/master/images/figures/repo.jpg)

## Структура пројекта
Овај репозиторијум садржи имплементације две поменуте методе у виду Python скрипти, као и три презентације у виду Jupyter свесака. Прва Jupyter свеска, 
[01 Neural Style Transfer.ipynb](https://github.com/qkolj/style-transfer/blob/master/01%20Neural%20Style%20Transfer.ipynb), објашњава и демонстрира методу Neural Style Transfer; друга Jupyter свеска, [02 P2-GAN.ipynb](https://github.com/qkolj/style-transfer/blob/master/02%20P2-GAN.ipynb), објашњава и демонстрира методу P²-GAN, док трећа Jupyter свеска, [03 Uporedni prikaz metoda.ipynb](https://github.com/qkolj/style-transfer/blob/master/03%20Uporedni%20prikaz%20metoda.ipynb), даје  неколико примера преноса стила користећи ове две методе над истим улазима. Како су обе ове методе хардверски захтевне, све ћелије Jupyter свесака су већ извршене и имају сачуван излаз. Уколико желите да самостално покренете ове Jupyter свеске, а немате одговарајући хардвер (довољно јаку графичку карту), можете их отворити на Google Colab платформи, која поседује GPU окружење, праћењем следећег [упутства](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb).

## Коришћење Python скрипти
Скрипт `nst.py` имплементира методу Neural Style Transfer, а `p2gan.py` имплементира методу P²-GAN. Помоћни скрипт `p2gan_models.py` одвојено имплементира моделе коришћене у `p2gan.py` ради боље читљивости кода. Оба скрипта, и `nst.py` и `p2gan.py`, имају детаљно упутство за коришћење које се може добити позивањем скрипте са аргументом `-h`. За тренирање P²-GAN методе потребно је скинути и отпаковати у овај репозиторијум [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) скуп података, док Neural Style Transfer имплементација сама из Python кода са сајта аутора рада скида VGG19 модел предтрениран на ImageNet скупу података.  

Пример позивања скрипте `nst.py` за рекреирање слике сличне оној на почетку овог текста:
```
$ python nst.py -c images/content/tubingen_256.jpg -s images/style/starry_night.jpg -i noise -d 256 343 --num-iters 5000 --beta 100
```

Пример позивања скриптa `p2gan.py` за рекреирање слике сличне оној на почетку овог текста:
```
$ python p2gan.py -m render -c images/content/tubingen_256.jpg -mp trained_models/starry_night.pth
```

Пример позивања скриптa `p2gan.py` ради тренирања модела `starry_night.pth`:
```
$ python p2gan.py -m train -s images/style/starry_night.jpg -sm trained_models/starry_night.pth -dp VOCdevkit/VOC2007/JPEGImages/ --lambda 0.005
```
