import numpy as np
import pandas as pd
import pymorphy2
from nltk.tokenize import word_tokenize
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import accuracy_score
from tok import word_tokenize


def tokenizer_norm(text):
    """
    форматирование строки https://github.com/kootenpv/tok/blob/master/README.md + pymorphy2
    возвращает предложение как список
    """
    morph = pymorphy2.MorphAnalyzer()
    result = word_tokenize(text)
    for i in range(len(result)):
        result[i] = morph.parse(result[i])[0]
        result[i] = result[i].normal_form

    result = [element for element in result if len(element) > 2]

    return result


def intersection(lst1, lst2):
    """
    нахождение пересечения двух списков
    """
    lst3 = [value for value in lst1 if value in lst2]
    return len(lst3)


def word_similarity(meaning_bts_tokenized, aim_word):
    """
    выбор толкования исходя из пересечения слов
    """
    df = pd.read_csv('train.csv', sep='\t', comment='#')

    labels = df[df['word'] == aim_word]['gold_sense_id']
    contexts = df[df['word'] == aim_word]['context']
    contexts_tokenized = list(map(tokenizer_norm, contexts))

    answers = pd.DataFrame(columns=["context", "meaning"])
    compare_intersections = np.zeros(2)

    for i in range(len(contexts_tokenized)):
        compare_intersections[0] = intersection(contexts_tokenized[i], meaning_bts_tokenized[0])
        compare_intersections[1] = intersection(contexts_tokenized[i], meaning_bts_tokenized[1])

        answers.loc[len(answers)] = [list(contexts)[i], np.argmax(compare_intersections) + 1]

        compare_intersections = np.zeros(2)

    acc = accuracy_score(list(answers['meaning']), list(labels))

    return answers, acc


def calc_mean_vector(text, embed_dict):
    """
    подсчет среднего вектора-представления для предложения
    :param embed_dict:
    :param text:
    :return:
    """
    vect = np.zeros(300, type, 'f')
    known_words_cnt = 0

    for element in text:
        if element in embed_dict.keys():
            vect += np.asarray(embed_dict[element].tolist())
            known_words_cnt += 1
    if known_words_cnt > 0:
        vect / float(known_words_cnt)

    return vect


def cosine_similarity(a, b):
    """
    вычисление косинусной близости между двумя словами
    :param a:
    :param b:
    :return:
    """
    a = np.asarray(a)
    b = np.asarray(b)

    return dot(a, b)/(norm(a)*norm(b))


def word2vec_similarity(meaning_bts_tokenized, aim_word, embeddings_dict):
    """
    выбор толкования исходя из пересечения слов
    """
    df = pd.read_csv('train.csv', sep='\t', comment='#')

    labels = df[df['word'] == aim_word]['gold_sense_id']
    contexts = df[df['word'] == aim_word]['context']
    contexts_tokenized = list(map(tokenizer_norm, contexts))

    texts = []  # средний вектор описаний из толкового словаря

    for meaning in meaning_bts_tokenized:
        texts.append(calc_mean_vector(meaning, embeddings_dict))

    answers = pd.DataFrame(columns=["context", "meaning"])
    compare_intersections = np.zeros(2)

    # проходим по каждому слову и выбираем подходящее для него толкование
    for i in range(len(contexts_tokenized)):
        for j in range(1):
            # compare_intersections[j] = intersection(contexts_tokenized[i], meaning_bts_tokenized[j])
            compare_intersections[j] = cosine_similarity(calc_mean_vector(contexts_tokenized[i], embeddings_dict),
                                                         calc_mean_vector(meaning_bts_tokenized[j], embeddings_dict))

        answers.loc[len(answers)] = [list(contexts)[i], np.argmax(compare_intersections) + 1]
        compare_intersections = np.zeros(2)

    acc = accuracy_score(list(answers['meaning']), list(labels))

    return answers, acc


def download_word_vectors(path):
    """
    загрузка обученных word embeddings нкря
    :param path: 
    :return: 
    """
    embeddings_dict = {}

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0].split("_")[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    return embeddings_dict


if __name__ == "__main__":

    path_to_model = "D:\\datasets\\языковые модели\\180\\model.txt"

    # ищем уникальные слова
    aim_words = ['замок', 'лук', 'бор', 'суда']

    # описание значений из словаря
    meaning_bts = [["Дворец и крепость феодала. Средневековый з. Развалины замка. 2. О дворцах, больших зданиях "
                   "затейливой архитектуры. Гатчинский з. Петровский з. 3. О тюрьме, остроге. Литовский з. Выборгский "
                   "з. <Замковый, -ая, -ое. ◊ Строить воздушные замки. Фантазировать, мечтать. Призрачные замки. "
                   "Беспочвенные фантазии, мечтания", "Устройство для запирания чего-л. ключом. Дверной з. Запереть "
                                                      "на з. Спрятать под з. (запереть, закрыть). Держать, "
                                                      "хранить под замком (держать запертым, взаперти). Быть на "
                                                      "замке; под замком (быть запертым). За тремя (семью, "
                                                      "десятью) замками (тщательно, надёжно спрятанный). Посадить на "
                                                      "з. (запереть, лишить свободы). // Приспособление для смыкания, "
                                                      "соединения концов или краёв чего-л. (цепочки, браслета и "
                                                      "т.п.). З. колье. Сломался з. молнии. 2. Спец. Устройство для "
                                                      "соединения подвижных частей механизмов. Пулемётный з. З. "
                                                      "браунинга. 3. Спец. Способ скрепления брёвен, брусьев и т.п. "
                                                      "частей деревянных конструкций, при котором они составляют как "
                                                      "бы одно целое. Срубить избу, дом в з. // Переплести пальцы, "
                                                      "плотно соединив их. Сложить руки в з. 4. Архит. Центральный "
                                                      "камень в вершине свода, арки. <Замковый, -ая, -ое (2-4 зн.). "
                                                      "Замочный, -ая, -ое (1 зн.). З-ая скважина."],
                   ["ЛУК, -а (-у); м. 1. Огородное растение сем. лилейных. Посадить лук лук пророс. 2. собирать. "
                    "Съедобные трубчатые листья или луковицы этого растения. Зелёный лук Репчатый лук Нарезать лука. "
                    "Посыпать луком. <Лучок, -чка (-чку); м. Ласк. Луковый, -ая, -ое. луковый запах. луковый соус. Горе"
                    " луковое. Шутл. О незадачливом, нерасторопном человеке.", "ЛУК, -а; м. Ручное оружие для метания стрел, изготовленное из гибкого, упругого стержня "
                    "(обычно деревянного), стянутого в дугу тетивой. Стрелять из лука. "
                    "Натянуть лук Вложить в лук стрелу."],
                   ["бор1, -а, предл. в бору, мн. -ы, -ов (лес); но (в названиях населенных пунктов) "
                    "Бор, -а, предл. в … Бору, напр.: Бор (город), Сосновый Бор (город), "
                    "Серебряный Бор (район в Москве), Красный Бор (поселок)", "бор, -а химический элемент; сверло; растение"],
                   ["Государственный орган, разбирающий гражданские споры и уголовные дела; помещение, в котором "
                    "находится такой орган. Гражданский с. Уголовный с. Подать в с. на кого-л. Приговор суда. Судья"
                    " районного суда. Народный с. (судебный орган первой инстанции). Верховный С. (в России: высший "
                    "судебный орган, осуществляющий надзор за деятельностью всех судебных органов). Военно-полевой,"
                    " полевой с. (судебный орган, рассматривающий преступления военных лиц, а также гражданских лиц "
                    "в военное время). Мировой с. (в России до 1917 г.: судебный орган, разбиравший мелкие "
                    "гражданские и уголовные дела). С. присяжных (судебный орган, состоящий из одного или "
                    "нескольких судей и коллегии присяжных заседателей). Пока с. да дело (пока решается или "
                    "совершается что-л.; о длительном, медленном процессе). Построили новый с. Остановка "
                    "автобуса у суда", "СУДНО, -а; мн. суда, -ов, -ам; ср. Плавучее сооружение, предназначенное для транспортных, "
                        "промысловых и военных целей, для научных исследований и т.п. Парусное с. Транспортные суда."
                        " Китобойное с. Морское, торговое с. Научно-исследовательское с. Плавать, служить на судне. "
                        "Судно-завод. Судно-рефрижератор. С. на воздушной подушке. Несамоходное с. "
                        "(буксируемое, парусное или гребное). ◊ Воздушное судно. Самолёт. <Судёнышко (см.). "
                        "Судовой, -ая, -ое. С. журнал. С. колокол. С-ая команда. С-ые машины. С-ая служба. "
                        "С-ые порядки. С-ая торговля."]]

    meaning_bts_tokenized = []

    for i in range(len(meaning_bts)):
        meaning_bts_tokenized.append([])
        for j in range(2):
            meaning_bts_tokenized[i].append([])
            meaning_bts_tokenized[i][j] = tokenizer_norm(meaning_bts[i][j])

    # загрузка языковой модели
    embed_dict = download_word_vectors(path_to_model)

    for i in range(len(aim_words)):
        answers, acc = word_similarity(meaning_bts_tokenized[i], aim_words[i])
        print("Исследуем слово ", aim_words[i])
        print("Точность для сравнения на основе пересечения слов  = ", acc)
        # print(answers)

        answers, acc = word2vec_similarity(meaning_bts_tokenized[i], aim_words[i], embed_dict)
        print("Точность для сравнения на основе word2vec  = ", acc)
        # print(answers)
        print("\n")












