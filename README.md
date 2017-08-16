# TopCoder-2017-07-Pathologic-Segmentation
## TopCoder competition Pathology Segmentation jule-august 2017

Jupyter-блокноты с результатами участия в соревновании [Pathology Segmentation](https://community.topcoder.com/tc?module=MatchDetails&rd=16950) на площадке **TopCoder** в июле-августе 2017.

В решении использовались пакеты: Python 2.7, OpenCV, Numpy, Keras, TensorFlow.

Директории содержат:
- [Jupyter-Net](Jupyter-Net) \- блокноты с программным кодом
- [Result](Result) \- результатирующие файлы в директориях с полученным счетом на LB
- [Data\-Keras/Optimizer](Data\-Keras/Optimizer) \- результаты по оптимайзерам (черновой вариант)

Блокноты имеют рабочее содержание и самодостаточны для воспроизведения при корректировке путей размещения данных.
Блокноты, которые использовались при получении результата, имеют номер счета в LB в заголовке.
Остальные блокноты имеют рабочее назначение.

Использовалась сеть UNet в стандартной модификации: 32-64-128-256-512-1024-..-32 и уменьшенной 32-..256-512-256-..-32
Использовались разные loss-функции в том числе и экзотические, в основном основанные на bce&dice
Использовалась плата GPU для работы GTX 970.

Полученное место: 26/63. Основная причина недостаточный опыт в задачах такого типа и ограниченное количество ресурсов (GPU) для получения
нормального результата в приемлиемое время. Необходимое количество эпох для данной схемы работ 
по оценкам других специалистов было необходимо в районе >200, что требовало для расчета на одном fold-е 4\*200=800 мин= 13 часов.
Поэтому общий расчет мог занять 13\*3 = 39 часов, что не является реалистичным.

Помещено для сохранения и возможного использования в будущем.
