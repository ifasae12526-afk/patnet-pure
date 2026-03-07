r""" Utility untuk custom dataset dengan XML annotation """
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def create_dataset_structure(root_path):
    """
    Buat struktur direktori untuk custom dataset
    
    Args:
        root_path: Path ke root direktori dataset
    """
    images_dir = os.path.join(root_path, 'images')
    annotations_dir = os.path.join(root_path, 'annotations')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    print(f"Dataset structure created at: {root_path}")
    print(f"  ├── images/")
    print(f"  │   ├── class1/")
    print(f"  │   │   ├── img1.jpg")
    print(f"  │   │   └── ...")
    print(f"  │   └── ...")
    print(f"  └── annotations/")
    print(f"      ├── class1/")
    print(f"      │   ├── img1.xml")
    print(f"      │   └── ...")
    print(f"      └── ...")


def create_sample_xml_annotation(output_path, filename, img_width=640, img_height=480, 
                                objects=None):
    """
    Buat sample XML annotation file (format Pascal VOC)
    
    Args:
        output_path: Path untuk output XML file
        filename: Nama file gambar
        img_width: Lebar gambar
        img_height: Tinggi gambar
        objects: List of dicts dengan format:
                 [{'name': 'class_name', 'xmin': 10, 'ymin': 20, 'xmax': 100, 'ymax': 150}, ...]
    
    Example:
        create_sample_xml_annotation(
            'data/custom/annotations/chicken/img1.xml',
            'img1.jpg',
            640, 480,
            [{'name': 'chicken', 'xmin': 100, 'ymin': 150, 'xmax': 400, 'ymax': 350}]
        )
    """
    root = ET.Element('annotation')
    
    # Add filename
    filename_elem = ET.SubElement(root, 'filename')
    filename_elem.text = filename
    
    # Add image size
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_width)
    height = ET.SubElement(size, 'height')
    height.text = str(img_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    
    # Add objects
    if objects:
        for obj in objects:
            obj_elem = ET.SubElement(root, 'object')
            
            name_elem = ET.SubElement(obj_elem, 'name')
            name_elem.text = obj.get('name', 'unknown')
            
            bndbox = ET.SubElement(obj_elem, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(obj.get('xmin', 0))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(obj.get('ymin', 0))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(obj.get('xmax', img_width))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(obj.get('ymax', img_height))
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"XML annotation created: {output_path}")


def create_polygon_xml_annotation(output_path, filename, img_width=640, img_height=480,
                                 objects=None):
    """
    Buat XML annotation dengan polygon segmentation
    
    Args:
        output_path: Path untuk output XML file
        filename: Nama file gambar
        img_width: Lebar gambar
        img_height: Tinggi gambar
        objects: List of dicts dengan format:
                 [{'name': 'class_name', 'points': [(x1,y1), (x2,y2), ...]}, ...]
    
    Example:
        create_polygon_xml_annotation(
            'data/custom/annotations/chicken/img1.xml',
            'img1.jpg',
            640, 480,
            [{'name': 'chicken', 'points': [(100,100), (300,100), (300,300), (100,300)]}]
        )
    """
    root = ET.Element('annotation')
    
    # Add filename
    filename_elem = ET.SubElement(root, 'filename')
    filename_elem.text = filename
    
    # Add image size
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_width)
    height = ET.SubElement(size, 'height')
    height.text = str(img_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    
    # Add objects with segmentation
    if objects:
        for obj in objects:
            obj_elem = ET.SubElement(root, 'object')
            
            name_elem = ET.SubElement(obj_elem, 'name')
            name_elem.text = obj.get('name', 'unknown')
            
            # Add segmentation if points provided
            if 'points' in obj:
                segmentation = ET.SubElement(obj_elem, 'segmentation')
                points_elem = ET.SubElement(segmentation, 'points')
                # Convert points to comma-separated string
                points = obj['points']
                points_str = ','.join([f"{x},{y}" for x, y in points])
                points_elem.text = points_str
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Polygon XML annotation created: {output_path}")


def validate_dataset_structure(dataset_path):
    """
    Validasi struktur custom dataset
    
    Args:
        dataset_path: Path ke root direktori dataset
    
    Returns:
        dict: Info tentang dataset
    """
    images_dir = os.path.join(dataset_path, 'images')
    annotations_dir = os.path.join(dataset_path, 'annotations')
    
    info = {
        'valid': True,
        'classes': [],
        'num_images': 0,
        'num_annotations': 0,
        'issues': []
    }
    
    if not os.path.exists(images_dir):
        info['valid'] = False
        info['issues'].append(f"Directory {images_dir} tidak ditemukan")
        return info
    
    if not os.path.exists(annotations_dir):
        info['valid'] = False
        info['issues'].append(f"Directory {annotations_dir} tidak ditemukan")
        return info
    
    # Check classes
    classes = [d for d in os.listdir(images_dir) 
              if os.path.isdir(os.path.join(images_dir, d))]
    info['classes'] = sorted(classes)
    
    # Count images and annotations
    for class_name in classes:
        class_img_dir = os.path.join(images_dir, class_name)
        class_anno_dir = os.path.join(annotations_dir, class_name)
        
        if os.path.exists(class_img_dir):
            img_files = [f for f in os.listdir(class_img_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            info['num_images'] += len(img_files)
        
        if os.path.exists(class_anno_dir):
            xml_files = [f for f in os.listdir(class_anno_dir) 
                        if f.lower().endswith('.xml')]
            info['num_annotations'] += len(xml_files)
        
        # Check if annotation exists for each image
        if os.path.exists(class_img_dir) and os.path.exists(class_anno_dir):
            img_files = [f for f in os.listdir(class_img_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            xml_files = set([f for f in os.listdir(class_anno_dir) 
                            if f.lower().endswith('.xml')])
            
            for img_file in img_files:
                img_base = os.path.splitext(img_file)[0]
                xml_name = img_base + '.xml'
                if xml_name not in xml_files:
                    info['issues'].append(
                        f"Missing annotation for {class_name}/{img_file}"
                    )
    
    print(f"\n=== Dataset Validation Report ===")
    print(f"Dataset Path: {dataset_path}")
    print(f"Status: {'✓ Valid' if info['valid'] else '✗ Invalid'}")
    print(f"Classes: {info['classes']}")
    print(f"Total Images: {info['num_images']}")
    print(f"Total Annotations: {info['num_annotations']}")
    
    if info['issues']:
        print(f"\nIssues found:")
        for issue in info['issues'][:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(info['issues']) > 10:
            print(f"  ... dan {len(info['issues']) - 10} issues lainnya")
    else:
        print("\n✓ No issues found!")
    
    return info


if __name__ == '__main__':
    # Example usage
    dataset_root = './data/custom_dataset'
    
    # Create directory structure
    create_dataset_structure(dataset_root)
    
    # Create sample annotations
    create_sample_xml_annotation(
        os.path.join(dataset_root, 'annotations', 'chicken', 'img1.xml'),
        'img1.jpg',
        640, 480,
        [{'name': 'chicken', 'xmin': 100, 'ymin': 150, 'xmax': 400, 'ymax': 350}]
    )
    
    # Validate dataset
    validate_dataset_structure(dataset_root)
