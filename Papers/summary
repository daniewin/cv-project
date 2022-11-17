# Summary of papers
## Camgöz: Joint End-to-End Sign Language Recognition and Translation, 2020

-	Spatio-temporal machine translation task
-	Multiple input channels:
    o	manual features (hand shape, movement, pose)
    o	non-manual features (facial expression, mouth, movement of head, shoulders, torso)
-	two ways:
    o	written language to sign video
    o	sign video to written language (two step pipeline)
        -	sign video -> (CV task, SLR) -> glosses -> (Language technology, SLT) -> written language
-	SLR challenges:
    o	Multiple channels
    o	Optional: Separate sentences and words in video -> automatic sign segmentation
    o	Continuous sign language recognition CSLR
    o	Use of direction and space
-	SLT challenges:
    o	Different word order
    o	One to many/many to one mapping
    o	Glosses include loss of information
-	General challenges:
    o	Lack of datasets
        -	Most studies until recent years used only weather forecast phoenix14T
        -	Recent years-> bigger datasets but not many studies on them yet
        -	With intermediate gloss step: performance depends only on SLR not SLT
-	End-to-end: combine SLR and SLT in one method
    o	Glosses as mid level representations still improve performance
        -	gloss input number is lower than frame number ->avoid long term dep issues
        -	without intermediate gloss supervision: translation task too difficult
-	Approach: 
    o	transformer encoder: SLR, decoder: SLT
    o	supervise with CTC loss for glosses
