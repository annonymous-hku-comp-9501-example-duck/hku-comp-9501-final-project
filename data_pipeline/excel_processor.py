import pandas as pd
import csv
import re
import yaml
import openai
import random


def excel_to_csv(excel_file, csv_file):
    try:
        # 使用pandas读取Excel文件
        df = pd.read_excel(excel_file)
        
        # 将数据保存为csv文件
        df.to_csv(csv_file, index=False)
        
        print("转换成功")
    except Exception as e:
        print(f"转换失败：{e}")

def chinese_english_dict(input: str, dict_path="data_pipeline/Chinese_English_dict.yaml") -> str:
    """
    Translate the Chinese string into English string based on 
    the ".yaml" dictionary file at dict_path.

    Args:
        input (str): The Chinese string that needs to be translated.
        dict_path (str): Path of dictionary file.
    """
    output = input
    with open(dict_path) as f:
        word_dict = yaml.load(f.read(),Loader=yaml.FullLoader)
    
    for chi,eng in word_dict.items():
        output=re.sub(chi,eng,output)
    return output   

def execel_translate_save(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_in, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)
        index=1

        # Skip the first row (header) in the input file
        next(reader, None)

        for row in reader:
            # Translate Chinese characters into English
            translate_row = [chinese_english_dict(cell) for idx,cell in enumerate(row)]
            writer.writerow(translate_row)

        
def excel_processor_func(excel_file,csv_origin_file,csv_trans_file):
    excel_to_csv(excel_file, csv_origin_file)
    execel_translate_save(csv_origin_file, csv_trans_file)


def gpt_refine(csv_trans_file, gpt_refine_file):
    openai.api_key = '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'
    prompt_list = [
        "I have a Q&A that I need you to help me modify and embellish, please make a few simple changes to the content in written language and keep the meaning same, you only need to answer the changes to: ",
        "I have a question and answer that I need you to help me embellish, please make a few simple changes to the content in written language and keep the meaning same, you only need to answer the changes to: ",
        "I have a question and answer that I need you to help me modify and embellish, please make a few simple changes to the content in written language and keep the meaning same, you only need to answer the changes to: ",
        "I have a Q&A that I need you to help me embellish, please make a few simple changes to the content in written language and keep the meaning same, you only need to answer the changes to: ",
        "Please help me to embellish the following question and answer with a few simple changes and keep the meaning same:"
        ]

    with open(csv_trans_file, 'r', newline='', encoding='utf-8') as csv_in, \
            open(gpt_refine_file, 'w', newline='', encoding='utf-8') as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)
        index = 1
        next(reader, None)

        for row in reader:
            # Translate Chinese characters into English
            origin_row = [cell for idx, cell in enumerate(row)]
            origin_question = origin_row[1]
            question_id = origin_question[:4].strip()
            todo_question = origin_question[4:].split('(')[0]
            #
            if question_id == '1.1':
                origin_answer = origin_row[2].strip()
                a_object_idx = '<' + origin_answer.split('<')[1].split('>')[0] + '>'
                todo_question = todo_question.replace('A', a_object_idx)
                todo_answer = origin_answer + ' ' + origin_row[3].strip()
            elif question_id == '1.2':
                a_object_idx = origin_row[2].strip()
                b_object_idx = '<' + origin_row[4].strip().split('<')[1].split('>')[0] + '>'
                todo_question = todo_question.replace('A', a_object_idx).replace('B', b_object_idx)
                todo_answer = a_object_idx + ' ' + origin_row[4].strip()
            elif question_id == '1.3':
                a_object_idx = origin_row[2].strip()
                todo_question = todo_question.replace('A', a_object_idx)
                todo_answer = a_object_idx + ' ' + origin_row[3].strip() + ' ' + origin_row[4].strip()
            elif question_id == '1.4':
                b_object_idx = '<' + origin_row[2].strip().split('<')[1].split('>')[0] + '>'
                a_object_idx = origin_row[3].strip()
                todo_question = todo_question.replace('A', a_object_idx).replace('B', b_object_idx)
                todo_answer = origin_row[2].strip() + ', ' + a_object_idx + ' ' + origin_row[4].strip() + origin_row[
                    5].strip()
            elif question_id == '1.5':
                a_object_idx = '<' + origin_row[2].strip().split('<')[1].split('>')[0] + '>'
                todo_question = todo_question.replace('A', a_object_idx)
                todo_answer = origin_row[2].strip() + ' ' + origin_row[3].strip()
            elif question_id == '1.6':
                b_object_idx = '<' + origin_row[2].strip().split('<')[1].split('>')[0] + '>'
                a_object_idx = '<' + origin_row[3].strip().split('<')[1].split('>')[0] + '>'
                todo_question = todo_question.replace('A', a_object_idx).replace('B', b_object_idx)
                todo_answer = origin_row[2].strip() + ', ' + origin_row[3].strip() + ' ' + origin_row[
                    4].strip() + ", " + origin_row[5].strip() + " " + origin_row[6].strip() + ", and " + origin_row[
                                  7].strip() + " " + origin_row[8].strip()
            elif question_id == '2.1':
                todo_answer = origin_row[2].strip() + ' ' + origin_row[3].strip()
            elif question_id == '2.2':
                todo_answer = origin_row[2].strip() + ' ' + origin_row[3].strip()
            elif question_id == '2.3':
                todo_answer = origin_row[2].strip() + ' ' + origin_row[3].strip()
            elif question_id == '2.4':
                todo_answer = origin_row[2].strip() + ' ' + origin_row[3].strip() + ", " + origin_row[
                    4].strip().replace('it', origin_row[3].strip()) + " " + origin_row[5].strip() + ', ' + origin_row[
                                  6].strip() + " " + origin_row[7].strip()
            elif question_id == '2.5':
                todo_answer = origin_row[2].strip() + " is: "
                if origin_row[3].strip() != 'none':
                    todo_answer = todo_answer + origin_row[3].strip() + ", "
                if origin_row[4].strip() != 'none':
                    todo_answer = todo_answer + origin_row[4].strip() + ", "
                if origin_row[5].strip() != 'none':
                    todo_answer = todo_answer + origin_row[5].strip() + ", "
                if origin_row[6].strip() != 'none':
                    todo_answer = todo_answer + origin_row[6].strip() + ", "
                if origin_row[7].strip() != 'none':
                    todo_answer = todo_answer + origin_row[7].strip() + ", "
                if origin_row[8].strip() != 'none':
                    todo_answer = todo_answer + origin_row[8].strip() + ", "
                todo_answer = todo_answer[:-2]
            elif question_id == '2.6':
                a_object_idx = '<' + origin_row[2].strip().split('<')[1].split('>')[0] + '>'
                todo_question = todo_question.replace('A', a_object_idx)
                todo_answer = origin_row[2].strip() + ', ' + origin_row[3].strip() + " " + origin_row[
                    4].strip() + ", " + origin_row[5].strip() + ' ' + origin_row[6].strip() + ", and " + origin_row[
                                  7].strip() + " " + origin_row[8].strip()
            elif question_id == '2.7':
                a_object_idx = '<' + origin_row[3].strip().split('<')[1].split('>')[0] + '>'
                b_object_idx = origin_row[2].strip().split('**')[1]
                todo_question = todo_question.replace('A', a_object_idx).replace('B', b_object_idx)
                todo_answer = origin_row[2].strip().split('**')[0] + b_object_idx + ', ' + origin_row[3].strip() + " " + \
                              origin_row[4].strip()
                if origin_row[4].strip() == 'high':
                    todo_answer = todo_answer + ", and "
                    if origin_row[6].strip() == 'yes':
                        todo_answer = todo_answer + 'there is a greater tendency for the responsibility to lie with the own vehicle'
                    else:
                        # todo_answer=todo_answer+'there is not a greater tendency for the responsibility to lie with the own vehicle'
                        todo_answer = todo_answer + 'there is a greater tendency for the responsibility to lie with other vehicles or external factors'

            todo_question = "Q: " + todo_question
            todo_answer = "A: " + todo_answer + '.'
            print(todo_question, todo_answer)

            messages = []
            system_message = "an English improver"
            messages.append({"role": "system", "content": system_message})
            # message = "I have a Q&A that I need you to help me modify and embellish, please make a few simple changes to the content in written language and keep the meaning same, you only need to answer the changes to: "+todo_question+" "+todo_answer
            message = random.choice(prompt_list) + todo_question + " " + todo_answer
            messages.append({"role": "user", "content": message})
            print(message)

            completion = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model='gpt-3.5-turbo-16k-0613',
                messages=messages,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
                timeout=1000,
            )
            return_message = completion.choices[0].message["content"]
            refine_question = return_message.split('A:')[0].strip()
            refine_answer = "A: " + return_message.split('A:')[1].strip()

            origin_row.append(refine_question)
            origin_row.append(refine_answer)

            writer.writerow(origin_row)

