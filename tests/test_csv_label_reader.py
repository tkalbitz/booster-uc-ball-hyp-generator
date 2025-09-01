"""Unit tests for csv_label_reader.py functions."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from src.csv_label_reader import read_csv_label, load_csv_collection


def create_temp_csv(content: str) -> Path:
    """Create a temporary CSV file with given content."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write(content)
    temp_file.close()
    return Path(temp_file.name)


def test_read_csv_label_basic() -> None:
    """Test basic functionality of read_csv_label."""
    csv_content = """image1_U.png;1;label1;Ball;10;20;30;40;Ball
image2_U.png;2;label2;Ball;50;60;70;80;Ball"""
    
    csv_file = create_temp_csv(csv_content)
    img_files = {
        'image1_U.png': '/path/to/image1_U.png',
        'image2_U.png': '/path/to/image2_U.png'
    }
    
    try:
        (imgs, labels, lines), skipped = read_csv_label(csv_file, img_files)
        
        assert len(imgs) == 2
        assert len(labels) == 2
        assert len(lines) == 2
        assert skipped == 0
        
        assert imgs[0] == '/path/to/image1_U.png'
        assert imgs[1] == '/path/to/image2_U.png'
        assert labels[0] == (10, 20, 30, 40)
        assert labels[1] == (50, 60, 70, 80)
        
    finally:
        csv_file.unlink()


def test_read_csv_label_coordinate_swapping() -> None:
    """Test that coordinates are swapped when x1 > x2 or y1 > y2."""
    csv_content = """image1_U.png;1;label1;Ball;30;40;10;20;Ball"""
    
    csv_file = create_temp_csv(csv_content)
    img_files = {'image1_U.png': '/path/to/image1_U.png'}
    
    try:
        (imgs, labels, lines), skipped = read_csv_label(csv_file, img_files)
        
        assert len(labels) == 1
        assert labels[0] == (10, 20, 30, 40)  # Coordinates should be swapped
        
    finally:
        csv_file.unlink()


def test_read_csv_label_ignore_small_boxes() -> None:
    """Test that small bounding boxes are ignored."""
    csv_content = """image1_U.png;1;label1;Ball;10;20;13;23;Ball
image2_U.png;2;label2;Ball;50;60;52;62;Ball"""  # Both boxes are < 4 pixels
    
    csv_file = create_temp_csv(csv_content)
    img_files = {
        'image1_U.png': '/path/to/image1_U.png',
        'image2_U.png': '/path/to/image2_U.png'
    }
    
    try:
        (imgs, labels, lines), skipped = read_csv_label(csv_file, img_files)
        
        assert len(imgs) == 0
        assert len(labels) == 0
        assert len(lines) == 0
        assert skipped == 2
        
    finally:
        csv_file.unlink()


def test_read_csv_label_ignore_missing_images() -> None:
    """Test that entries for missing image files are ignored."""
    csv_content = """image1_U.png;1;label1;Ball;10;20;30;40;Ball
missing_image_U.png;2;label2;Ball;50;60;70;80;Ball"""
    
    csv_file = create_temp_csv(csv_content)
    img_files = {'image1_U.png': '/path/to/image1_U.png'}  # missing_image_U.png not in dict
    
    try:
        with patch('src.csv_label_reader._logger') as mock_logger:
            (imgs, labels, lines), skipped = read_csv_label(csv_file, img_files)
        
        assert len(imgs) == 1
        assert len(labels) == 1  
        assert len(lines) == 1
        assert skipped == 0
        
        # Verify warning was logged
        mock_logger.warning.assert_called_once_with("Skip missing image file %s", "missing_image_U.png")
        
    finally:
        csv_file.unlink()


def test_read_csv_label_ignore_pattern() -> None:
    """Test that ignore pattern removes all entries for a file."""
    csv_content = """image1_U.png;1;label1;Ball;10;20;30;40;Ball
image1_U.png;2;label2;Ball;50;60;70;80;Ball
image2_U.png;3;label3;Ignore
image2_U.png;4;label4;Ball;90;100;110;120;Ball"""
    
    csv_file = create_temp_csv(csv_content)
    img_files = {
        'image1_U.png': '/path/to/image1_U.png',
        'image2_U.png': '/path/to/image2_U.png'
    }
    
    try:
        (imgs, labels, lines), skipped = read_csv_label(csv_file, img_files)
        
        # Since image1 appears twice, the second occurrence gets skipped due to duplicate handling
        # image2 gets ignored completely due to Ignore pattern
        # So we expect no results
        assert len(imgs) == 0
        assert len(labels) == 0
        assert len(lines) == 0
        assert skipped == 2  # One from duplicate handling, one from ignore
        
    finally:
        csv_file.unlink()


def test_read_csv_label_duplicate_handling() -> None:
    """Test handling of duplicate image entries - both entries get removed."""
    csv_content = """image1_U.png;1;label1;Ball;10;20;30;40;Ball
image1_U.png;2;label2;Ball;50;60;70;80;Ball"""
    
    csv_file = create_temp_csv(csv_content)
    img_files = {'image1_U.png': '/path/to/image1_U.png'}
    
    try:
        (imgs, labels, lines), skipped = read_csv_label(csv_file, img_files)
        
        # Due to the duplicate handling logic, when duplicate is found, 
        # the first entry gets removed and second gets skipped
        assert len(imgs) == 0
        assert len(labels) == 0
        assert len(lines) == 0
        assert skipped == 1  # Second entry was skipped due to duplicate
        
    finally:
        csv_file.unlink()


def test_read_csv_label_non_matching_lines() -> None:
    """Test that non-matching lines are ignored."""
    csv_content = """image1_U.png;1;label1;Ball;10;20;30;40;Ball
invalid line format
another;invalid;line
image2_U.png;2;label2;Ball;50;60;70;80;Ball"""
    
    csv_file = create_temp_csv(csv_content)
    img_files = {
        'image1_U.png': '/path/to/image1_U.png',
        'image2_U.png': '/path/to/image2_U.png'
    }
    
    try:
        (imgs, labels, lines), skipped = read_csv_label(csv_file, img_files)
        
        assert len(imgs) == 2
        assert len(labels) == 2
        assert len(lines) == 2
        assert skipped == 0
        
    finally:
        csv_file.unlink()


def test_read_csv_label_empty_file() -> None:
    """Test behavior with empty CSV file."""
    csv_content = ""
    
    csv_file = create_temp_csv(csv_content)
    img_files = {}
    
    try:
        (imgs, labels, lines), skipped = read_csv_label(csv_file, img_files)
        
        assert len(imgs) == 0
        assert len(labels) == 0
        assert len(lines) == 0
        assert skipped == 0
        
    finally:
        csv_file.unlink()


def test_read_csv_label_mixed_valid_size() -> None:
    """Test mix of valid and invalid size bounding boxes."""
    csv_content = """image1_U.png;1;label1;Ball;10;20;30;40;Ball
image2_U.png;2;label2;Ball;50;60;53;63;Ball
image3_U.png;3;label3;Ball;100;110;120;130;Ball"""
    
    csv_file = create_temp_csv(csv_content)
    img_files = {
        'image1_U.png': '/path/to/image1_U.png',
        'image2_U.png': '/path/to/image2_U.png',
        'image3_U.png': '/path/to/image3_U.png'
    }
    
    try:
        (imgs, labels, lines), skipped = read_csv_label(csv_file, img_files)
        
        # Only image1 and image3 should be valid (size >= 4)
        assert len(imgs) == 2
        assert len(labels) == 2
        assert len(lines) == 2
        assert skipped == 1  # image2 was skipped due to small size
        
        assert '/path/to/image1_U.png' in imgs
        assert '/path/to/image3_U.png' in imgs
        assert (10, 20, 30, 40) in labels
        assert (100, 110, 120, 130) in labels
        
    finally:
        csv_file.unlink()


def test_load_csv_collection() -> None:
    """Test load_csv_collection function."""
    # Create CSV files
    csv1_content = """image1_U.png;1;label1;Ball;10;20;30;40;Ball"""
    csv2_content = """image2_U.png;2;label2;Ball;50;60;70;80;Ball"""
    
    csv1_file = create_temp_csv(csv1_content)
    csv2_file = create_temp_csv(csv2_content)
    
    # Create collection file
    collection_content = f"""{csv1_file.name}
{csv2_file.name}"""
    collection_file = create_temp_csv(collection_content)
    
    img_files = {
        'image1_U.png': '/path/to/image1_U.png',
        'image2_U.png': '/path/to/image2_U.png'
    }
    
    try:
        imgs, labels, skipped = load_csv_collection(collection_file, img_files)
        
        assert len(imgs) == 2
        assert len(labels) == 2
        assert skipped == 0
        
        assert '/path/to/image1_U.png' in imgs
        assert '/path/to/image2_U.png' in imgs
        assert (10, 20, 30, 40) in labels
        assert (50, 60, 70, 80) in labels
        
    finally:
        csv1_file.unlink()
        csv2_file.unlink()
        collection_file.unlink()


def test_load_csv_collection_with_skipped() -> None:
    """Test load_csv_collection with some skipped entries."""
    # Create CSV files with mixed content
    csv1_content = """image1_U.png;1;label1;Ball;10;20;30;40;Ball"""
    csv2_content = """image2_U.png;2;label2;Ball;50;60;52;62;Ball"""  # Too small
    
    csv1_file = create_temp_csv(csv1_content)
    csv2_file = create_temp_csv(csv2_content)
    
    collection_content = f"""{csv1_file.name}
{csv2_file.name}"""
    collection_file = create_temp_csv(collection_content)
    
    img_files = {
        'image1_U.png': '/path/to/image1_U.png',
        'image2_U.png': '/path/to/image2_U.png'
    }
    
    try:
        imgs, labels, skipped = load_csv_collection(collection_file, img_files)
        
        assert len(imgs) == 1  # Only image1 is valid
        assert len(labels) == 1
        assert skipped == 1  # image2 was skipped
        
        assert imgs[0] == '/path/to/image1_U.png'
        assert labels[0] == (10, 20, 30, 40)
        
    finally:
        csv1_file.unlink()
        csv2_file.unlink()
        collection_file.unlink()