# 2DUnet-trail
Unet2d for brats segmentation.

to use this repositories code, you should edit code as your need.
in this project, the Unet2d is mainly on single modal segmentaion; to change into multi modals seg, there are serval places need to edit.

In train19.py (240,250*3) should change to (240,250*6) for 3 modals,and range(1) should change to range(3) for size aligning.
In dataloader19_brats, "if 'flair' in f :    # if is data or 't1' in f or 't1ce' in f or 't2' in f "->"if 'flair' in f or 't1' in f or 't1ce' in f or 't2' in f"
