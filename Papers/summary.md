# Summary of papers
## Camgöz: Joint End-to-End Sign Language Recognition and Translation, 2020

-	Spatio-temporal machine translation task 
-	Multiple input channels: 
    -	manual features (hand shape, movement, pose) 
    -	non-manual features (facial expression, mouth, movement of head, shoulders, torso) 
-	two ways: 
    -	written language to sign video 
    -	sign video to written language (two step pipeline)
        - sign video -> (CV task, SLR) -> glosses -> (Language technology, SLT) -> written language
-	SLR challenges:
    -	Multiple channels
    -	Optional: Separate sentences and words in video -> automatic sign segmentation
    -	Continuous sign language recognition CSLR vs Isolated sign language recognition
    -	Use of direction and space
-	SLT challenges:
    -	Different word order -> non-monotonic
    -	One to many/many to one mapping
    -	Glosses include loss of information
-	General challenges:
    -	Lack of datasets
        -	Most studies until recent years used only weather forecast phoenix14T
        -	Recent years-> bigger datasets but not many studies on them yet
    -	With intermediate gloss step: performance depends only on SLR not SLT (bottleneck)
    -	Without intermediate gloss supervision step: translation task too difficult
-	End-to-end: combine SLR and SLT in one method
    -	Glosses as mid level representations still improve performance
        -	gloss input number is lower than frame number ->avoid long term dep issues
-	Approach: 
    -	transformer encoder: SLR, decoder: SLT
    -	supervise with CTC loss for glosses
    -	multiple co-dependent transformers, simultaneously trained to jointly solve related tasks
    -	seq-to-seq task: ! frames are many more than glosses or text
- Evaluation:
    -
- Results:
    -
    
