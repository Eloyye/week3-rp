import requests

def main():
    URL = 'http://127.0.0.1:8000/uploadfile/'
    input_file = 'inputs/math_expression.pdf'
    with open(input_file, 'rb') as file:
        file = {'request_file': file}
        resp = requests.post(url=URL, files=file)
        print(resp.json())

if __name__ == '__main__':
    main()