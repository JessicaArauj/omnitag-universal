from types import SimpleNamespace

from package.python import tools


def test_load_dataset_parses_csv(tmp_path):
    csv_path = tmp_path / 'dataset.csv'
    csv_path.write_text('id,user,rating,text\n1,u,7.5,Great water\n', encoding='latin1')

    analyser = tools.RankingAnalyser(gpt=SimpleNamespace())
    analyser.load_dataset(str(csv_path))

    assert analyser.dataset[0]['text'] == 'Great water'
    assert analyser.dataset[0]['rating'] == 7.5


def test_get_texts_formats_dataset():
    analyser = tools.RankingAnalyser(gpt=SimpleNamespace())
    analyser.dataset = [{'text': 'Sample answer', 'rating': 6.25}]

    texts = analyser._get_texts()

    assert texts == ['rating = 6.2\ntext = Sample answer']
