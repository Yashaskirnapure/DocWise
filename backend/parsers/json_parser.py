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


def get_description(images, text, image_text, context_window=2):
    image_captions = {}

    text_ids = sorted(text.keys(), key=lambda x: int(x) if str(x).isdigit() else float('inf'))

    for img_id in images:
        caption = image_text.get(img_id, {}).get('content', '')
        img_path = images[img_id]['reference']
        img_desc = ""

        try:
            idx = text_ids.index(img_id)
        except ValueError:
            idx = -1

        if idx != -1:
            before = text_ids[max(0, idx - context_window): idx]
            after = text_ids[idx + 1: idx + 1 + context_window]

            for tid in before + after:
                img_desc += " " + text[tid].get('content', '')

        image_captions[img_id] = {
            "filePath": img_path,
            "image_text": caption,
            "description": img_desc.strip()
        }

    return image_captions
