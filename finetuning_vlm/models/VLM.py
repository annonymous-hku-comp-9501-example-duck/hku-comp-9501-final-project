import torch
import torch.nn as nn
import logging

from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator


logger = logging.getLogger(__name__)

    
class VLM(nn.Module):
    def __init__(
            self,
            name="Salesforce/blip2-flan-t5-xl",
            use_pretrained=True,
            finetuning= 'qformer', #lora, qformer
            freeze=False,
            **kwargs
        ):
        self.model_type = name
        super(VLM, self).__init__()

        # dummy_accelerator = Accelerator()
        # current_device = dummy_accelerator.process_index

        if use_pretrained:
            # self.model = AutoModel.from_pretrained(self.model_type)
            self.processor = AutoProcessor.from_pretrained(self.model_type)
            # Let's define the LoraConfig
            if finetuning == 'lora':
                # about 8bit and device_map='auto': https://github.com/huggingface/transformers/issues/21736
                model = Blip2ForConditionalGeneration.from_pretrained(self.model_type) #, device_map={'':current_device}, load_in_8bit=True)
                config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    # target_modules=["query", "key"]
                )

                self.model = get_peft_model(model, config)
                self.model.print_trainable_parameters()
            elif finetuning == 'qformer':
                model = Blip2ForConditionalGeneration.from_pretrained(self.model_type)
                model.qformer.training = True
                for param in model.language_model.parameters():
                    param.requires_grad = False
                for param in model.vision_model.parameters():
                    param.requires_grad = False
                # for param in model.qformer.parameters():
                #     try:
                #         param.requires_grad = True
                #         # print('success')
                #     except:
                #         pass

                # for name, mod in model.named_modules():
                #     if 'vision' in name or 'language_model' in name:
                #         for param in mod.parameters():
                #             param.requires_grad = False
                #     elif 'qformer' in name:
                #         for param in mod.parameters():
                #             try:
                #                 param.requires_grad = True
                #             except:
                #                 pass
                #         print(name)
                    # print(name)

                self.model = model
                # self.model.print_trainable_parameters()

        else:
            raise NotImplementedError
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        
    def forward(self, question_input_ids, answers_input_ids, pixel_values, return_string=False):
        # change dtype of pixel_values to float16

        
        if return_string:
            self.model.eval()
            # output_string1 = self.processor.batch_decode(outputs.logits.argmax(-1))
            generated_ids = self.model.generate(
                input_ids=question_input_ids,
                pixel_values=pixel_values,
                # attention_mask=batch["attention_mask"].to(device),
                num_beams=4,
                max_length=64,
                early_stopping=True,
                # num_return_sequences=5,

            )
            output_string2 = []
            for i, generated_id in enumerate(generated_ids):
                # Decode text
                output_string2.append(self.processor.tokenizer.decode(generated_id, skip_special_tokens=True))

            self.model.train()
            return output_string2, generated_ids
        
        else:
            outputs = self.model(
                input_ids=question_input_ids,
                pixel_values=pixel_values,
                labels=answers_input_ids
                )
        
        return outputs
    

    
if __name__=="__main__":
    model = VLM()
    pass