# pip install openai-whisper
import whisper
import torch

def speech_recognition(model='base'):
    speech_model = whisper.load_model(model)
    torch.cuda.init()
    with torch.cuda.device('cuda'):
        result = speech_model.transcribe('C:/Users/user/PycharmProjects/pythonProject1/data/netstalk.mp3', language='ru')

    with open(f'transcription_{model}.txt', 'w', encoding="utf-8") as file:
        file.write(result['text'])


def main():
    models = {1: 'tiny', 2: 'base', 3: 'small', 4: 'medium', 5: 'large'}

    for k, v in models.items():
        print(f'{k}:{v}')

    model = int(input('Выберите модель передав цифру от 1 до 5: '))

    if model not in models.keys():
        raise KeyError(f'Модели {model} нет в списке!')

    print('Запущен процесс транскрибации, пожалуйста ожидайте...')
    speech_recognition(model=models[model])


if __name__ == '__main__':
    main()
