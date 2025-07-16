import json

def normalize_node(element):
    obj_id = element.get('ObjectID')
    text = element.get('Text')
    path = element.get('Path', '')
    file_paths = element.get('filePaths')

    if 'Figure' in path:
        if file_paths:
            return {
                'id': obj_id,
                'type': 'figure',
                'reference': file_paths[0]
            }
        else:
            return {
                'id': obj_id,
                'type': 'image-text',
                'content': text
            }

    return {
        'id': obj_id,
        'type': 'text',
        'content': text
    }

def parse_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    elements = data.get("elements")

    images = {}
    text = {}
    image_text = {}

    for elem in elements:
        if 'Table' in elem.get('Path'):
            continue
        else:
            normalized_element = normalize_node(elem)
            id = normalized_element['id']
            if normalized_element['type'] == 'figure':
                images[id] = normalized_element
            elif normalized_element['type'] == 'text':
                text[id] = normalized_element
            else:
                image_text[id] = normalized_element

    return { 'images': images, 'text': text, 'image_text': image_text }


def get_description(images, text, image_text):
    image_captions = {}

    available_text_ids = set(text.keys())

    for img_id in images:
        img_txt = image_text.get(img_id, {}).get("content", "")
        text_desc = ""

        prev_id = img_id - 1
        while prev_id not in available_text_ids and prev_id >= 0:
            prev_id -= 1
        if prev_id in text:
            text_desc += text[prev_id]['content'] + " "

        next_id = img_id + 1
        while next_id not in available_text_ids and next_id <= max(available_text_ids):
            next_id += 1
        if next_id in text:
            text_desc += text[next_id]['content']

        image_captions[img_id] = {
            "filePath": images[img_id]["reference"],
            "image_text": img_txt,
            "description": text_desc.strip()
        }

    return image_captions