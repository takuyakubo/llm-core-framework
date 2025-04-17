"""
Prompt management module.

This module provides tools for managing prompts across different LLM providers.
It includes utilities for defining provider-specific prompts, extracting and
setting variables, and handling attachments like images.

Example usage:
    from core.prompts.managers import PromptManager
    from langchain_core.messages import SystemMessage, HumanMessage
    from core.llm.providers import ProviderType
    
    # Create a prompt manager
    analyze_image_prompt = PromptManager("analyze_image_prompt", description="Image analysis prompt")
    
    # Define provider-specific prompts
    analyze_image_prompt[ProviderType.ANTHROPIC.value] = [
        SystemMessage(content="You are an image analysis assistant."),
        HumanMessage(content="Analyze the following image and describe what you see: {image_description}")
    ]
    
    analyze_image_prompt[ProviderType.OPENAI.value] = [
        SystemMessage(content="Analyze images in detail."),
        HumanMessage(content="Please look at this image and provide a detailed analysis: {image_description}")
    ]
    
    # Attach image support
    analyze_image_prompt.append_attach_key("image_data")
    
    # Use the prompt
    formatted_prompt = analyze_image_prompt({
        "image_description": "This is a photo of a mountain landscape.",
        "_attach_image_data": {"type": "image", "data": image_b64}
    })
    
    # Send to model
    response = model.invoke(formatted_prompt)
"""

import logging
from copy import deepcopy
from string import Formatter
from typing import Self

from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


def extract_variables_from(format_string):
    """
    Extract variable names from a format string.
    
    This function extracts the names of all format variables in a string
    using Python's string.Formatter.
    
    Args:
        format_string (str): The format string to extract variables from
        
    Returns:
        list: A list of variable names found in the format string
        
    Example:
        >>> extract_variables_from("Hello, {name}! You are {age} years old.")
        ['name', 'age']
        
    フォーマット文字列から変数名を抽出します。
    
    この関数は、Pythonのstring.Formatterを使用して、文字列内のすべての
    フォーマット変数の名前を抽出します。
    """
    formatter = Formatter()
    variables = []
    for _, field_name, _, _ in formatter.parse(format_string):
        if field_name is not None:
            variables.append(field_name)
    return variables


def extract_vars(target, kws):
    """
    Recursively extract all variables from a template object.
    
    This function walks through a template object (which may be a list, dict,
    or BaseMessage) and extracts all format variables from strings.
    
    Args:
        target: The template object to extract variables from
        kws (list): A list to store the extracted variable names
        
    Returns:
        list: The updated list of variable names
        
    テンプレートオブジェクトからすべての変数を再帰的に抽出します。
    
    この関数は、テンプレートオブジェクト（リスト、辞書、またはBaseMessageの
    場合があります）を通過し、文字列からすべてのフォーマット変数を抽出します。
    """
    if isinstance(target, list):
        for v in target:
            extract_vars(v, kws)
    elif isinstance(target, dict):
        for k in target:
            extract_vars(target[k], kws)
    elif isinstance(target, BaseMessage):
        extract_vars(target.content, kws)
    elif isinstance(target, str):
        kws += extract_variables_from(target)
    return kws


def assign_vars(target, kws):
    """
    Recursively assign values to variables in a template object.
    
    This function walks through a template object and formats all strings
    with the provided keyword arguments.
    
    Args:
        target: The template object to format
        kws (dict): A dictionary of variable names and their values
        
    Returns:
        The formatted template object
        
    テンプレートオブジェクト内の変数に値を再帰的に割り当てます。
    
    この関数は、テンプレートオブジェクトを通過し、提供されたキーワード引数で
    すべての文字列をフォーマットします。
    """
    if isinstance(target, list):
        return [assign_vars(v, kws) for v in target]
    elif isinstance(target, dict):
        return {k: assign_vars(v, kws) for k, v in target.items()}
    elif isinstance(target, BaseMessage):
        return type(target)(assign_vars(target.content, kws))
    elif isinstance(target, str):
        return target.format(**kws)
    return target


class PromptManager:
    """
    Manager for provider-specific prompt templates.
    
    This class provides a way to define and use prompts that are tailored
    to different LLM providers while maintaining a consistent interface.
    It supports variable substitution and attachments like images.
    
    Attributes:
        prompt_name (str): The name of the prompt template
        prompt_description (str): A description of the prompt's purpose
        prompt_contents (dict): A dictionary mapping provider keys to templates
        variables (list): A list of variables used in the templates
        default_key (str): The default provider key to use
        attach_prefix (str): Prefix for attachment variables
        
    プロバイダー固有のプロンプトテンプレートのマネージャー。
    
    このクラスは、一貫したインターフェースを維持しながら、異なるLLM
    プロバイダーに合わせたプロンプトを定義および使用する方法を提供します。
    変数置換や画像などの添付ファイルをサポートしています。
    """

    def __init__(self, prompt_name, description="", use_default=True) -> None:
        """
        Initialize a prompt manager.
        
        Args:
            prompt_name (str): The name of the prompt template
            description (str, optional): A description of the prompt's purpose
            use_default (bool, optional): Whether to use the default provider
                                         if a requested provider is not found
                                         
        プロンプトマネージャーを初期化します。
        
        引数：
            prompt_name (str): プロンプトテンプレートの名前
            description (str, optional): プロンプトの目的の説明
            use_default (bool, optional): リクエストされたプロバイダーが見つからない場合に
                                         デフォルトのプロバイダーを使用するかどうか
        """
        self.prompt_name = prompt_name
        self.prompt_description = description
        self.prompt_contents = dict()
        self.variables = []
        self.default_key = None
        self.get_item_logic = lambda x: x
        self.use_default = use_default
        self.attach_prefix = (
            "_attach_"  # DSLで_attach_　とついたkeyには添付で対応する。
        )

    def __setitem__(self, key, value) -> None:
        """
        Set a prompt template for a specific provider key.
        
        This method adds or updates a prompt template for the specified provider.
        It also extracts and validates the variables used in the template.
        
        Args:
            key (str): The provider key
            value: The prompt template
            
        Raises:
            Exception: If the variables in the new template don't match existing ones
            
        特定のプロバイダーキーのプロンプトテンプレートを設定します。
        
        このメソッドは、指定されたプロバイダーのプロンプトテンプレートを追加または
        更新します。また、テンプレートで使用される変数を抽出して検証します。
        """
        variables = extract_vars(value, [])
        if self.default_key is None:
            self.default_key = key
            self.variables = variables
        else:
            if set(self.variables) != set(variables):
                raise Exception(
                    "新しく設定するテンプレートは元のテンプレートと同一のformat変数を持たなくてはいけません。"
                )
        self.prompt_contents[key] = value

    def __getitem__(self, key: str) -> Self:
        """
        Get a prompt manager for a specific provider key.
        
        This method returns the prompt manager with the default key set to the
        specified provider key. If the key is not found and use_default is True,
        it falls back to the default key.
        
        Args:
            key (str): The provider key
            
        Returns:
            PromptManager: The prompt manager with the default key set
            
        Raises:
            Exception: If the key is not found and use_default is False
            
        特定のプロバイダーキーのプロンプトマネージャーを取得します。
        
        このメソッドは、デフォルトキーが指定されたプロバイダーキーに設定された
        プロンプトマネージャーを返します。キーが見つからず、use_defaultがTrueの場合、
        デフォルトキーにフォールバックします。
        """
        key_ = self.get_item_logic(key)
        if key_ not in self.prompt_contents:
            if self.use_default:
                logger.warning(
                    f"{self.prompt_name}に対するkeyで想定外のものが呼び出されました。expected in: {list(self.prompt_contents.keys())}, actual: {key} -> {key_}"
                )
                return self
            else:
                raise Exception(
                    f"{self.prompt_name}に対するkeyは次のうちいずれかにして下さい: {list(self.prompt_contents.keys())}"
                )
        self.default_key = key_
        return self

    def __call__(self, kwargs):
        """
        Format the prompt template with the provided variables.
        
        This method formats the prompt template for the default provider
        with the provided variables and attachments.
        
        Args:
            kwargs (dict): A dictionary of variable names and their values
            
        Returns:
            ChatPromptTemplate: The formatted prompt template
            
        Raises:
            Exception: If required variables are missing
            
        Example:
            >>> prompt = prompt_manager({
            ...     "image_description": "A landscape photo",
            ...     "_attach_image_data": {"type": "image", "data": image_b64}
            ... })
            >>> response = model.invoke(prompt)
            
        提供された変数でプロンプトテンプレートをフォーマットします。
        
        このメソッドは、デフォルトプロバイダーのプロンプトテンプレートを
        提供された変数と添付ファイルでフォーマットします。
        """
        kws = kwargs.keys()
        if not (set(self.variables) <= set(kws)):
            raise Exception(
                f"{self.prompt_name}の呼び出しは、あらかじめ決められた引数が必要です。expected: {self.variables}, actual: {kws}"
            )
        prompt_content = deepcopy(self.prompt_contents[self.default_key])
        prompt_content = assign_vars(prompt_content, kwargs)
        attached_contents = []
        for k in kws:
            if k.startswith(self.attach_prefix):
                attached_contents = self.attach(kwargs[k], attached_contents)
        if attached_contents:
            prompt_content += [HumanMessage(content=attached_contents)]
        return ChatPromptTemplate(prompt_content)

    @staticmethod
    def attach(image_info, content_list):
        """
        Attach media content to a prompt.
        
        This method handles attaching media content like images to a prompt.
        It supports both single items and lists of items.
        
        Args:
            image_info: The media content to attach (dict or list of dicts)
            content_list (list): The existing list of attached content
            
        Returns:
            list: The updated list of attached content
            
        Raises:
            ValueError: If the attachment is not a dict or list
            
        Example:
            >>> content_list = []
            >>> image_info = {"type": "image", "data": image_b64}
            >>> content_list = PromptManager.attach(image_info, content_list)
            
        プロンプトにメディアコンテンツを添付します。
        
        このメソッドは、画像などのメディアコンテンツをプロンプトに添付する処理を行います。
        単一アイテムとアイテムのリストの両方をサポートしています。
        """
        if isinstance(image_info, list):
            content_list += image_info
        elif isinstance(image_info, dict):
            content_list += [image_info]
        else:
            raise ValueError("添付できるタイプはlistかdictのみです。")
        return content_list

    def append_attach_key(self, key: str):
        """
        Add an attachment key to the list of variables.
        
        This method adds an attachment variable to the prompt manager.
        Attachment variables are used to include media content like images
        in the prompt.
        
        Args:
            key (str): The base key name (without the attach prefix)
            
        Example:
            >>> prompt_manager.append_attach_key("image_data")
            >>> # Now you can use "_attach_image_data" when formatting the prompt
            
        添付キーを変数リストに追加します。
        
        このメソッドは、添付変数をプロンプトマネージャーに追加します。
        添付変数は、画像などのメディアコンテンツをプロンプトに含めるために使用されます。
        """
        self.variables += [self.attach_prefix + key]
