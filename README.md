
# Image Retrieval Using Textual Cues

## Description

- Image Retrieval Using Textual Cues, to retrieve the relevant images based on the text information available in the image rather than the objects available in it
- Here the process is to localize and recognize the text, and then query the database, as in a text retrieval problem.
- This application does not rely on an exact localization and recognition pipeline.
- This application follows a query-driven search approach, which requires finding the approximate locations of characters in the text query.
- The key idea behind the proposed scheme is optical character recognition module.
- This module will detect the textual information from the query image and all the images in the image database.
- To find the similarity between textual information extracted from the query image and the textual information extracted from the image database we use the longest common substring algorithm
## Input and output

Give Input as the image based on which you need to select similar images.
The output will be relavent images.

```bash
  Input: Query image
  Output: Relavent Images based on textual cues in query image
```


## Tech Stack

**Tech Stack:** Python, MySQL, Pytesseract OCR, OpenCV, numpy, pandas

