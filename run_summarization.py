import models.extractive_models, models.abstractive_models
import os
import warnings
import warnings
import math
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, TextSplitter

#Disable specific warning of SKLEARN because it is not relevant and also not fixable
warnings.filterwarnings('ignore', category=FutureWarning, message='^The default value of `n_init` will change from 10 to \'auto\' in 1.4')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ExtractiveSummarizationModel:

    model_classes = ['RoBERTa', 'LegalBERT']

    def __init__(self, model_type):

        self.model_type = model_type
        self.model, self.tokenizer = self.load_extractive_model()

    
    def load_extractive_model(self):
        """
        Loads the extractive model and its tokenizer based on the specified model type.
        
        Returns:
            model: The loaded extractive model.
            tokenizer: The loaded tokenizer for the model.
        Raises:
            ValueError: If an invalid extractive model type is specified.
        """
        if self.model_type not in self.model_classes:
            raise ValueError(f"Invalid extractive model type: {self.model_type}\nPlease select from {self.model_classes}")    
        
        self.model, self.tokenizer = models.extractive_models.select_extractive_model(self.model_type)
        
        print(f"Succesfully loaded {self.model_type} model and its tokenizer")
        
        return self.model, self.tokenizer


    def summarize(self, text, extractive_compression_ratio):
        """
        Summarizes the given text using the model.

        Args:
            text (str): The input text to be summarized.

        Returns:
            str: The extractive summary of the input text.
        """

        summary = self.model(text, ratio = extractive_compression_ratio)

        #print(f"Extractive summary: \n----------------------\n{summary}\n----------------------\n")

        return summary


class AbstractiveSummarizationModel:
    
    model_classes = ['BART', 'T5']

    def __init__(self, model_type):
        self.model_type = model_type
        self.model, self.tokenizer = self.load_abstractive_model()


    def load_abstractive_model(self):
        """
        Loads the abstractive model and its tokenizer based on the specified model type.

        Returns:
            model (object): The loaded abstractive model.
            tokenizer (object): The tokenizer associated with the loaded model.

        Raises:
            ValueError: If an invalid abstractive model type is specified.
        """
        if self.model_type not in self.model_classes:
            raise ValueError(f"Invalid abstractive model type: {self.model_type}\nPlease select from {self.model_classes}")
        
        self.model, self.tokenizer = models.abstractive_models.select_abstractive_model(self.model_type)

        print(f"Succesfully loaded {self.model_type} model and its tokenizer")

        return self.model, self.tokenizer


    def summarize(self, text):
            """
            Summarizes the input text using the loaded abstractive model.
            
            Args:
                text (str): The input text to be summarized.
            
            Returns:
                str: The summarized text.
            """
            #TODO: Check if max_length is correct
            inputs = self.tokenizer([text], max_length= self.tokenizer.model_max_length, return_tensors='pt', truncation=True)

            # Generate the summarized text
            summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=250, early_stopping=True)     
            summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces = False)[0]

            return summary
    
    
    def train(self, train_data, val_data, epochs, batch_size, learning_rate, warmup_steps, weight_decay, output_dir):
        """
        Trains the model using the provided training data.

        Args:
            train_data (list): The training data.
            val_data (list): The validation data.
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            learning_rate (float): The learning rate for training.
            warmup_steps (int): The number of warmup steps for training.
            weight_decay (float): The weight decay for training.
            output_dir (str): The output directory to save the trained model.

        Returns:
            None
        """
        pass
    

class SummarizationPipeline:
    def __init__(self, extractive_model_type, abstractive_model_type, extractive_compression_ratio = 0.5):
        self.extractive_model = ExtractiveSummarizationModel(extractive_model_type)
        self.abstractive_model = AbstractiveSummarizationModel(abstractive_model_type)
        self.extractive_compression_ratio = extractive_compression_ratio

    def summarize(self, text):

        extractive_steps_required = self.calculate_amount_of_extractive_steps(text)
        print(f"Extractive steps required: {extractive_steps_required}")

        if extractive_steps_required >= 1:
            for _ in range(extractive_steps_required):

                chunks = self.get_text_chunks(text)
                print(f"Amount of chunks: {len(chunks)}")
                intermediary_summary = ""
                for chunk in chunks:
                    chunk_summary = self.extractive_model.summarize(chunk, self.extractive_compression_ratio)
                    intermediary_summary += chunk_summary
                
                text = intermediary_summary

            abstractive_summary = self.abstractive_model.summarize(text)

        else:
            abstractive_summary = self.abstractive_model.summarize(text)

        #TODO: Concatenate all summaries of chunks to one summary to be used as input for abstractive summarization

        
        return abstractive_summary


    def calculate_amount_of_extractive_steps(self, text):
        """
        Calculates the amount of extractive steps needed to compress the given text before it can bed to the abstractive summarization model.

        Parameters:
        text (str): The input text to be compressed.

        Returns:
        int: The amount of extractive steps needed.
        """

        amount_of_tokens = len(self.extractive_model.tokenizer.tokenize(text))
        print(f"Amount of tokens in text: {amount_of_tokens}")
        
        context_length = self.abstractive_model.tokenizer.model_max_length

        #TODO: Check if 1 still applies. This is just a placeholder for now.
        variable = 1
        outcome = (math.log10((variable*context_length)/amount_of_tokens))/(math.log10(self.extractive_compression_ratio))

        amount_of_extractive_steps = math.floor(outcome)

        return amount_of_extractive_steps
    

    def get_text_chunks(self, text):
        """
        Takes raw text and returns text chunks.

        Parameters:
        text (str): String of text from document.

        Returns:
        chunks (list): List of chunks (str) of 505 tokens.
        """ 
        print(f"Max length of tokenizer: {self.extractive_model.tokenizer.model_max_length}")
        #TODO: chunk_overlap is hardcoded for now. This should be a parameter in the future.
        #TODO: Chunk_overlap is subtracted from chunk_size. This is not correct but TokenTextSplitter will make chunks too big if chunk_overlap is not subtracted from chunk_size.
        
        text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.extractive_model.tokenizer, 
            chunk_size=self.extractive_model.tokenizer.model_max_length - 50,
            chunk_overlap=50)
        
        chunks = text_splitter.split_text(text)

        return chunks
        

if __name__ == "__main__":

    pipeline = SummarizationPipeline(extractive_model_type = 'RoBERTa', abstractive_model_type='BART')

    text = "The HIC Rotterdam is one of the largest fuels and chemicals clusters in the world. It processes ~50 megatons of crude oil per year into fuels and chemical products,which are then transported around the world. In line withthe Paris Climate Agreement, the Port of Rotterdam (POR) has set the ambition to reach net-zero CO2 emissions by 2050. This requires realizing an energy and feedstock transition to replace fossil-based fuels and feedstocks with sustainable alternatives.The energy transition at the HIC has gained momentum, and plans are being executed to achieve emissions reduction targets (e.g. energy supply and infrastructure for renewable electricity, green hydrogen, carbon capture, etc.). Strong policies, mandates and subsidy support are available to achieve these climate targets. Recent energy security concerns have further increased the urgency to reduce dependency on fossil fuels and move towards sustainable alternatives. Scope of the challenge: The transition from crude oil to more sustainable feedstocks will be an extensive process. It involves transforming the fuels and chemicals value chain and the interconnected system of large-scale assets that has been built over the past century. This feedstock transition is still in its early stages, and many challenges need to be overcome before this can be achieved. For one, sustainable alternatives are less efficient to convert to useful end-products. They also require more energy and space, along with major new investments in supply chain infrastructures and assets. The regulatory and legislative framework to support investments and business cases is still emerging. For example, no ETS system, scope 3 targets or taxonomy for circular feedstocks yet exist. Existing mandates currently prioritize the available sustainable carbon feedstocks for use in low-carbon fuels instead of in products with a longer life cycle, such as plastics where there are no definite mandates. Ultimately, if the current trend continues, there will likely be strong global competition for securing sustainable carbon feedstock to replace all the crude oil we use today. Securing access to scarce green carbon molecules will be vital. Preparations and roadmap Companies in the HIC Rotterdam are aware of the opportunities and risks of being a front-runner in the feedstock transition. Some companies are taking initial steps and are starting to integrate new/sustainable feedstocks in their processes. Although the required technology largely exists, it is currently mostly deployed at relatively small scale. The share of sustainable feedstock today represents less than 10 percent of total feedstock flows4 in the HIC Rotterdam. There is not yet a clear picture of the target feedstock mix nor the speed at which we want (and need) to achieve it. Preparations must start early and new coalitions need to be formed to establish the new sustainable value chains. This paper is intended as a starting point for further action and to inform the key choices to be made regarding: — the target ambition level and the speed at which we want to achieve it; — which sustainable feedstocks to prioritize and for which applications; — what to do locally in the HIC Rotterdam and what to import from other locations; — how to fit it all given space and energy system constraints; and — how to retain a vital industry in a global competitive market during the transition. Bio oils are made from second-generation biomass, used cooking oil (UCO), residual vegetable oil, residual animal fats or other biogenic waste streams. They can be processed into biofuels directly or used as co-feed in refineries. These days they can be used in most oil processing installations and are already used to produce biodiesel, SAF and bio-naphtha. Demand for these products is expected to grow significantly due to tightening EU regulations and mandates, such as REDII. Pyrolysis oil is produced by the slow heating of plastic (waste) material, which is currently incinerated or landfilled. It can be used to replace crude oil as a (circular) co-feed in existing refineries and petrochemical sites. Pyrolysis oil is liquid, making it a highly suitable match for most oil process installations. Converting plastic waste into pyrolysis oil requires significant heat and power. The first projects for producing pyrolysis oil for fuels have been announced, and a cluster is being developed at the HIC Rotterdam. However, pyrolysis oil is expected to be largely imported from other locations where the solid waste is collected and processed into pyrolysis oil. This process requires significant heat, power, space, as well as the sourcing of plastic waste, which is highly decentralized. Green methanol is produced from green hydrogen and biogenic or atmospheric carbon. Gray methanol is already used as a fuel additive and in the production of various fuels and chemicals (e.g. aviation and shipping fuels, MTBE, aromatics and olefins). In addition to its end-use flexibility, green methanol can be synthesized from green hydrogen and multiple carbon feedstocks: — Residual gas is a by-product from industrial processes (CO2, CO, CH4, H2) and is currently used primarily as a combustion fuel (e.g. in process furnaces and steam boilers). Residual gas can be used as a carbon feedstock to produce green methanol instead of as a fuel (for which green hydrogen or electricity can be used). An estimated 5– 10 Mtpa is available at the HIC today that could potentially be redeployed. The main source of carbon feedstock at the HIC now comes from crude oil processing, yet this can shift in the future towards green carbon sources with the increased processing of biogenic, circular and captured carbon. To justify the required investments, policy and taxonomy changes must recognize the carbon emission savings. — Solid biomass is 2nd generation material that does not compete with food production and is gasified to produce biogenic syngas. This process requires significant amounts of space, hydrogen and power. Biogenic carbon will ultimately be a limited resource globally. — Atmospheric carbon (direct air capture (DAC)) can be captured from the atmosphere and as such is essentially an unlimited resource. DAC technology is still very costly and consumes large amounts of energy and space. Significant cost breakthroughs are required to make it commercially viable. Hydrogen from renewable electricity sources will play a key role in decarbonizing the HIC Rotterdam. Green hydrogen and its derivatives (e.g. ammonia) can replace fossil fuels and feedstocks as a source of energy in making selected products, i.e., those that do not contain carbon molecules, such as fertilizers. It is also a key input for converting sustainable carbon feedstocks into useful end products. Whether used to hydro-process bio oil feedstocks or to convert CO2 into syngas (CO + H2) for further support in FisherTropsch synthesis or methanol synthesis processes, green hydrogen will be required in large quantities to enable the feedstock transition. Chapter 3: Compatibility. While liquids (bio- and pyrolysis oils) are easiest to implement, all new feedstocks will require major changes to the fuels and chemicals value chains. Asset base requirements The HIC’s current asset base is highly geared towards processing liquid and gaseous hydrocarbons. The cluster’s existing distillation and separation columns, gas treatment equipment, gas and liquid heat exchangers, pumps and compressors are most compatible with sustainable feedstocks that have similar physical and chemical properties. In comparison, solid feedstocks – especially biomass – are much less compatible with the current fuels and chemicals value chains and require much higher investments in infrastructure and pre-processing assets. The extent of asset transformation needed for sites focused on producing fuels may be very different from that for sites which mainly produce chemicals. For example, fuel production sites might choose to focus on maximizing their existing hydrotreatment assets to upgrade sustainable feedstock, while chemical sites may opt for upstream integration in sourcing naphtha-like feedstocks so that they can continue using their downstream infrastructure (perhaps including their cracker infrastructure). The compatibilities of the different feedstocks with the existing assets are described below. Connecting the feedstocks and the supporting energy flows with the processing assets will require a transformation of the infrastructure as well. The current fossil-focused infrastructure and integrated fuels and petrochemicals set-up will shift to multiple sustainable carbon feedstocks that will be imported on a large scale. This requires major investments in new supply chain infrastructure to be able to import, store and distribute different liquid and gaseous fuels and feedstocks. Feedstock compatibilities Each feedstock will have their own specific fit within the respective fuels and chemicals value chains: — Refining will need a mix of low-carbon (hydrogenbased) feedstocks for processing industry and road transport, as well as sustainable carbon-based 12 Compatibility 3 While liquids (bio- and pyrolysis oils) are easiest to implement, all new feedstocks will require major changes to the fuels and chemicals value chains. Recycling Refineries & fuel production Bio- / E-Naphta End of life DAC CCU CCU Captured Carbon CO² Bio-based Carbon Imported Circular & Captured Carbon Hydrogen Fuel use Fossil Carbon Petchem Circular Carbon Fuel use End of life CO²² Petchem Fossil Carbon Bio-based Carbon Hydrogen Refineries & fuel production Residual gases are by-products of existing industrial processes in the HIC and need to be converted to useful syngas before they can be used as a feedstock. This requires substantial investments and new asset and conversion infrastructure, particularly when the energy content in these gases is very low. Higher CO2 content requires more hydrogen-intensive conversions to convert it into syngas (CO plus H2). This needs to be followed by Fisher-Tropsch or methanol synthesis units to convert the syngas into carbon products that meet the requirement of having an energy density comparable to that of crude oil. In addition, if residual gas is to be used as a feedstock source, investments will be needed to switch the fuel supply (from residual gas to e.g. hydrogen or electricity). feedstocks that be converted into fuels suitable for shipping and aviation. — Chemicals can include sustainable and circular carbon pathways that are able to produce pure base chemicals such as olefins and aromatics. Therefore, these value chains are expected to focus on different pathways and have their own timelines. This may ultimately lead to the decoupling of refining and chemicals activities. feedstocks that be converted into fuels suitable for shipping and aviation. — Chemicals can include sustainable and circular carbon pathways that are able to produce pure base chemicals such as olefins and aromatics. Therefore, these value chains are expected to focus on different pathways and have their own timelines. This may ultimately lead to the decoupling of refining and chemicals activities. Bio oils and pyrolysis oil can be implemented relatively easily with the existing refinery assets and infrastructure. Some, such as vegetable oils, are already co-processed in hydrotreating and -cracking units in some of the HIC refineries. These feedstocks are highly compatible with the current refinery asset base, and securing them at large scale should therefore be prioritized. Chemical plants, however, will require additional hydrotreating units to pre-process these feedstocks, limiting direct compatibility with this value chain. Plastic waste and biomass are solid feedstocks and require significant investments in pre-processing units that produce high(er) energy-density liquid intermediate feedstocks before they can be converted into useful end products. For plastic waste, there are synergies with existing waste handling companies, and the required investments across the fuels and chemicals value chain are expected to be modest. Biomass, on the other hand, will require much higher investments in pre-processing. Removing the water content and other impurities (e.g. oxygenates) from biomass will involve gasifiers and liquefaction- and/or fermentation units. Green methanoleq and its derivatives lie somewhere in between the solids and the other liquid feedstocks in terms of asset compatibility. It shares the benefits of bio oils and pyrolysis oil in being a liquid feedstock, which is an advantage in terms of storage, transport and general product handling. Its main challenge, however, lies in the fact that methanol-to-product conversion requires specific chemistry, making it significantly different than the refining of crude oil and also limiting its compatibility with traditional refining units. However, for specific chemical products, methanol can offer unique flexibility: once the methanol has been converted in e.g., olefins and aromatics, downstream chemical units–which already use these molecules as a feedstock–can still be used."

    summary = pipeline.summarize(text)
    
    print(f"Final summary is:\n------------------------\n{summary}")
    