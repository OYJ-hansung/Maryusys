Topic:
Finger_Transferlearning - Tensorflow.js transfer learning using mobilenet and denseLayer

About:
기본적으로 Finger_Transferlearning 예제와 Pacman_Transferlearning 예제가 동일해서 두 개를 비교 분석 했습니다.

	A. Difference

		1. Pacman은 node.js위에서 동작을 하고, Finger은 그냥 자바스크립트에서 동작을 합니다.

		2. Pacman은 index.html과 ui.js와 같은 document파일과 index.js파일이 복잡하게 꼬여 있어서 모델의 전이학습과
		관련된 여러가지 파라미터 값들을 여기저기서 불러왔습니다.

		3. Finger은 필수적인 ui를 html과 css파일에서 아주 간단하게 정의를 했고, 그 외의 전이학습과 관련된
		모든 내부기능을 index.js에서 정의를 했습니다.

	B. Common

		1. 사용되는 pre-trained model(https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json)이 동일하고
		 truncate과 transfer learning하는 과정이 동일합니다.


Result:
정상적으로 잘 동작합니다.