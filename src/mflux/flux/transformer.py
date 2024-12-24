import mlx.core as mx
import mlx.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention optimisée pour MLX"""

    def __init__(self, num_heads, dims):
        super().__init__()
        self.num_heads = num_heads
        self.dims = dims
        self.head_dim = dims // num_heads

        # Projections linéaires
        scale = 1 / mx.sqrt(mx.array([self.head_dim], dtype=mx.float32))
        self.scale = scale

        # Initialisation optimisée des poids
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.o_proj = nn.Linear(dims, dims)

        # Compiler les opérations fréquentes
        self._reshape_qkv = mx.compile(self._reshape_qkv_fn)
        self._compute_attention = mx.compile(self._compute_attention_fn)

        # Initialiser les paramètres
        mx.eval(self.parameters())

    def _reshape_qkv_fn(self, x, batch_size, seq_len):
        """Reshape les tenseurs Q, K, V de manière optimisée"""
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(0, 2, 1, 3)
        return x

    def _compute_attention_fn(self, q, k, v):
        """Calcule l'attention de manière optimisée"""
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)
        return out

    def __call__(self, query, key, value):
        """Calcule l'attention multi-tête de manière optimisée"""
        try:
            batch_size = query.shape[0]
            seq_len = query.shape[1]

            # Projections linéaires
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            # Reshape pour les têtes d'attention
            q = self._reshape_qkv(q, batch_size, seq_len)
            k = self._reshape_qkv(k, batch_size, seq_len)
            v = self._reshape_qkv(v, batch_size, seq_len)

            # Calcul de l'attention
            out = self._compute_attention(q, k, v)

            # Reshape et projection finale
            out = out.transpose(0, 2, 1, 3)
            out = out.reshape(batch_size, seq_len, self.dims)
            out = self.o_proj(out)

            return out

        except Exception as e:
            print(f"Erreur lors de l'attention: {str(e)}")
            return None


class TransformerBlock(nn.Module):
    """Bloc de transformateur optimisé pour MLX"""

    def __init__(self, hidden_size, intermediate_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads

        # Couches d'attention
        self.self_attn = MultiHeadAttention(
            num_heads=num_attention_heads,
            dims=hidden_size,
        )

        # Couches feed-forward avec initialisation optimisée
        self.mlp_linear1 = nn.Linear(hidden_size, intermediate_size)
        self.mlp_act = nn.GELU()
        self.mlp_linear2 = nn.Linear(intermediate_size, hidden_size)

        # Normalisation avec epsilon optimisé pour float32
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)

        # Compiler les opérations fréquentes
        self._forward = mx.compile(self._forward_fn)

        # Initialiser les paramètres
        mx.eval(self.parameters())

    def _forward_fn(self, hidden_states):
        """Forward pass optimisé"""
        # Attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, hidden_states, hidden_states)
        if hidden_states is None:
            return None
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp_linear1(hidden_states)
        hidden_states = self.mlp_act(hidden_states)
        hidden_states = self.mlp_linear2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def __call__(self, hidden_states):
        """Applique le bloc de transformateur de manière optimisée"""
        try:
            return self._forward(hidden_states)
        except Exception as e:
            print(f"Erreur dans le bloc de transformateur: {str(e)}")
            return None


class Transformer(nn.Module):
    """Transformateur optimisé pour MLX sure Mac"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config

        # Configuration optimisée pour Apple Silicon
        self.hidden_size = 1024
        self.intermediate_size = 4096
        self.num_attention_heads = 16
        self.num_layers = 8

        # Projection des latents avec initialisation optimisée
        self.latent_proj = nn.Linear(4 * 64 * 64, self.hidden_size)

        # Couches de transformateur
        self.layers = [
            TransformerBlock(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_attention_heads=self.num_attention_heads,
            )
            for _ in range(self.num_layers)
        ]

        # Projection finale
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.final_proj = nn.Linear(self.hidden_size, 4 * 64 * 64)

        # Compiler les opérations fréquentes
        self._forward = mx.compile(self._forward_fn)

        # Initialiser les paramètres
        mx.eval(self.parameters())

    def _forward_fn(self, hidden_states):
        """Forward pass optimisé"""
        batch_size = hidden_states.shape[0]

        # Projeter les latents
        hidden_states = hidden_states.reshape(batch_size, -1)
        hidden_states = self.latent_proj(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, 1, -1)

        # Appliquer les couches
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if hidden_states is None:
                return None

        # Projection finale
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, -1)
        hidden_states = self.final_proj(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, 4, 64, 64)

        return hidden_states

    def __call__(
        self,
        t: int,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        hidden_states: mx.array,
        config=None,
    ) -> mx.array:
        """Prédit le bruit de manière optimisée"""
        try:
            config = config or self.config
            if config is None:
                print("Attention: Aucune configuration n'est disponible")
                return None

            return self._forward(hidden_states)

        except Exception as e:
            print(f"Erreur lors de la prédiction: {str(e)}")
            return None

    def trainable_parameters(self) -> dict:
        """Retourne les paramètres entraînables du modèle"""
        params = {}

        # Ajouter les paramètres de la projection des latents
        latent_params = self.latent_proj.parameters()
        if latent_params:
            for name, value in latent_params.items():
                if isinstance(value, mx.array):
                    params[f"latent_proj.{name}"] = value.astype(mx.float32)

        # Ajouter les paramètres des couches de transformateur
        for i, layer in enumerate(self.layers):
            layer_params = layer.parameters()
            if layer_params:
                for name, value in layer_params.items():
                    if isinstance(value, mx.array):
                        params[f"layer_{i}.{name}"] = value.astype(mx.float32)

        # Ajouter les paramètres de la projection finale
        final_norm_params = self.final_layer_norm.parameters()
        if final_norm_params:
            for name, value in final_norm_params.items():
                if isinstance(value, mx.array):
                    params[f"final_layer_norm.{name}"] = value.astype(mx.float32)

        final_proj_params = self.final_proj.parameters()
        if final_proj_params:
            for name, value in final_proj_params.items():
                if isinstance(value, mx.array):
                    params[f"final_proj.{name}"] = value.astype(mx.float32)

        if not params:
            print("Attention: Aucun paramètre entraînable trouvé dans le transformateur")
            return {}

        return params
